# ============================================================================
# 這支腳本的功能：在 TPU 上，用 PyTorch/XLA 微調 Stable Diffusion 的 UNet。
#
# 跟一般在 GPU 上寫 PyTorch 訓練迴圈最大的不同，是 TPU 走的是「延遲執行
# （lazy execution）+ 圖編譯（graph compilation）」的模式：
#   - 在 GPU 上，你寫的每一行 tensor 運算（如 loss.backward()）幾乎是「馬上執行」
#     （eager mode），出錯或印數值都能立刻看到。
#   - 在 TPU/XLA 上，你寫的運算會先被記錄成一張計算圖，直到程式呼叫
#     xm.mark_step()（或某些會強制同步的操作）時，才會把整張圖丟給 XLA
#     編譯器編譯成機器碼、送到 TPU 上執行。這代表：
#       1. 第一次執行某個 shape/流程時會有「編譯時間」，通常比後面每一步都慢很多
#          （可對照 README 中 inference 範例：compile time 720 秒 vs 之後每次 1.8 秒）。
#       2. 如果程式中頻繁印出 tensor 數值（例如每一步都印 loss），會打斷延遲執行的
#          批次優化，逼著 XLA 提前把圖跑完，訓練速度會明顯變慢（對照下面
#          --print_loss 的說明）。
#       3. 多卡（多顆 TPU 核心）之間的資料平行，不是用 GPU 常見的 NCCL +
#          DistributedDataParallel（每張卡各自跑一份模型、算完梯度後 all-reduce），
#          而是用 XLA 的 GSPMD／SPMD：只寫「單一份」模型與訓練邏輯，透過
#          xs.get_1d_mesh()、xs.set_global_mesh()、ShardingSpec 告訴編譯器
#          「這個 batch 維度要怎麼切到哪些裝置上」，實際的資料切分、通訊、同步
#          都交給 XLA 編譯器自動處理。
# ============================================================================

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm  # TPU/XLA 版的「裝置與同步」工具，對應 GPU 上的 torch.cuda
import torch_xla.debug.profiler as xp  # XLA 專用的效能分析器（trace），不是 GPU 常用的 torch.profiler / Nsight
import torch_xla.distributed.parallel_loader as pl  # 把 CPU 端的 DataLoader 包裝成會自動預先搬資料到 TPU 的版本
import torch_xla.distributed.spmd as xs  # GSPMD／SPMD：TPU 上「切分張量到多裝置」的核心工具
import torch_xla.runtime as xr  # TPU 執行環境資訊（有幾顆裝置、有幾台 host 等）
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, HfArgumentParser

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
from utils import save_model_card


if is_wandb_available():
    pass

# PROFILE_DIR / CACHE_DIR 是從環境變數讀進來的，對應 README 裡 export 的那兩個變數
PROFILE_DIR = os.environ.get("PROFILE_DIR", None)
CACHE_DIR = os.environ.get("CACHE_DIR", None)
if CACHE_DIR:
    # TPU/XLA 特有：把「編譯好的計算圖」快取到硬碟。
    # 因為 XLA 每次遇到新的 shape 或流程都要重新編譯（很慢，見上方說明），
    # 如果把編譯結果快取下來，下次啟動同樣的程式就能直接讀取，省掉重複編譯的時間。
    # GPU 上的一般 PyTorch 訓練沒有這個「圖編譯快取」的概念（torch.compile 也有類似快取，但非必要）。
    xr.initialize_cache(CACHE_DIR, readonly=False)

# TPU/XLA 特有：開啟 SPMD（Single Program Multiple Data）模式。
# 開啟後，所有裝置（TPU 核心）會執行「同一份」程式，資料如何切分到各裝置
# 是由後面的 mesh + ShardingSpec 決定，而不是像 GPU DDP 那樣每個 process 各跑一份獨立行程。
xr.use_spmd()

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}
PORT = 9012  # XLA profiler server 要監聽的埠號


class TrainSD:
    """封裝訓練迴圈的類別：把訓練需要的模型元件、optimizer、dataloader 都存起來，
    然後由 start_training() 跑固定步數的訓練，step_fn() 則是「單一步」在做的事。
    """

    def __init__(
        self,
        vae,
        weight_dtype,
        device,
        noise_scheduler,
        unet,
        optimizer,
        text_encoder,
        dataloader,
        args,
    ):
        self.vae = vae
        self.weight_dtype = weight_dtype
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.optimizer = optimizer
        self.text_encoder = text_encoder
        self.args = args
        # TPU/XLA 特有：取得目前的裝置網格（mesh），描述了「有哪些 TPU 裝置、要怎麼排列」，
        # 之後 SPMD 會依照這個 mesh 把 batch 資料切分到各裝置。GPU 版訓練沒有這個概念，
        # 通常是用 torch.distributed 的 rank/world_size 來手動切分資料。
        self.mesh = xs.get_global_mesh()
        self.dataloader = iter(dataloader)
        self.global_step = 0

    def run_optimizer(self):
        self.optimizer.step()

    def start_training(self):
        dataloader_exception = False
        measure_start_step = args.measure_start_step
        assert measure_start_step < self.args.max_train_steps
        total_time = 0
        # 主訓練迴圈：跑 max_train_steps 步
        for step in range(0, self.args.max_train_steps):
            try:
                # 從 dataloader 拿下一批資料。因為下面用的是 MpDeviceLoader（會預先把資料搬到 TPU），
                # 這裡拿到的 batch 其實已經在 TPU 裝置上了，不需要像 GPU 常見寫法那樣手動 .to(device)。
                batch = next(self.dataloader)
            except Exception as e:
                dataloader_exception = True
                print(e)
                break
            if step == measure_start_step and PROFILE_DIR is not None:
                # TPU/XLA 特有：xm.wait_device_ops() 會阻塞，直到目前所有已排入佇列、
                # 尚未執行完的裝置運算都跑完為止。因為 XLA 是非同步執行（丟出運算後不會馬上等結果），
                # 若要準確計時（量測「平均每步花多少時間」），就必須先確定前面的運算已經真的執行完，
                # 否則計時器會量到「排隊等待」而非「實際運算」的時間。這點類似 GPU 上的
                # torch.cuda.synchronize()，但在 XLA 這裡幾乎是必要動作，因為預設就是非同步/延遲執行。
                xm.wait_device_ops()
                # 啟動 XLA profiler，把接下來 profile_duration 毫秒內的 trace 存到 PROFILE_DIR，
                # 之後可以用 TensorBoard 之類的工具打開分析。這是 XLA 專屬的效能分析流程，
                # 跟 GPU 常用的 torch.profiler / Nsight Systems 是不同的工具鏈。
                xp.trace_detached(f"localhost:{PORT}", PROFILE_DIR, duration_ms=args.profile_duration)
                last_time = time.time()
            # 真正跑一步訓練（forward + backward + optimizer step），細節見 step_fn()
            loss = self.step_fn(batch["pixel_values"], batch["input_ids"])
            self.global_step += 1

            def print_loss_closure(step, loss):
                print(f"Step: {step}, Loss: {loss}")

            if args.print_loss:
                # TPU/XLA 特有：這裡不是直接 print(loss)，而是用 xm.add_step_closure()
                # 註冊一個「等這一步真正執行完之後才會被呼叫」的函式。
                # 原因：loss 此刻只是計算圖裡的一個「延遲張量（lazy tensor）」，還沒有真正算出數值；
                # 如果直接存取 loss 的數值（例如直接 print(loss)），會強迫 XLA 立刻把目前累積的整張圖
                # 編譯並執行（等於提前呼叫一次同步），這樣會打斷原本「多步驟一起打包執行」的優化，
                # 讓每一步的訓練速度變慢。add_step_closure 可以把「印出數值」這件事延後到
                # 該步驟真的執行完之後才做，盡量不影響效能。這是 GPU 上 print(loss.item()) 不會遇到的問題
                # （GPU eager 模式下每一步本來就是立刻執行）。
                xm.add_step_closure(
                    print_loss_closure,
                    args=(
                        self.global_step,
                        loss,
                    ),
                )
        # TPU/XLA 特有：迴圈跑完後，用 mark_step() 明確告訴 XLA「把目前累積、還沒執行的計算圖
        # 送去編譯並執行」。在 XLA 的延遲執行模型裡，如果不主動呼叫 mark_step()，
        # 有些尚未被讀取數值的運算可能一直停留在圖裡沒有真的被執行。
        xm.mark_step()
        if not dataloader_exception:
            # 再次等待所有裝置運算跑完，確保計時準確（理由同上方 wait_device_ops 的說明）
            xm.wait_device_ops()
            total_time = time.time() - last_time
            print(f"Average step time: {total_time / (self.args.max_train_steps - measure_start_step)}")
        else:
            print("dataloader exception happen, skip result")
            return

    def step_fn(
        self,
        pixel_values,
        input_ids,
    ):
        """單一訓練步驟：這就是 Stable Diffusion（DDPM 系列）訓練的標準流程：
        1. 用 VAE 把圖片壓縮成 latent（潛在空間）
        2. 隨機選一個時間步 t，加上對應強度的雜訊，得到「加噪後的 latent」
        3. 用文字 encoder 把 caption 轉成 embedding，餵給 UNet 一起預測雜訊（或 v-prediction）
        4. 算預測值跟「真正加的雜訊」之間的 MSE loss，反向傳播、更新 UNet 參數
        （VAE 和 text_encoder 全程凍結，不會被更新，這點在 README 也有提到）

        xp.Trace(...) 是 XLA profiler 的區塊標記，用來在 trace 檔案裡標出
        「forward / backward / optimizer_step 各花多少時間」，方便之後用工具分析效能瓶頸，
        跟 GPU 常見的 torch.profiler.record_function 用途類似，但屬於 XLA 專用工具。
        """
        with xp.Trace("model.forward"):
            self.optimizer.zero_grad()
            # VAE encode：把像素圖片壓縮成 latent，並乘上 scaling_factor（VAE 訓練時的慣例）
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            # 產生和 latents 同樣形狀的隨機高斯雜訊，這是 diffusion model 要學著「預測」的目標
            noise = torch.randn_like(latents).to(self.device, dtype=self.weight_dtype)
            bsz = latents.shape[0]
            # 幫 batch 裡每一筆資料隨機抽一個 diffusion 時間步 t（雜訊強度）
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # 依照時間步 t，把雜訊加到乾淨的 latent 上，得到 UNet 實際看到的輸入
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            # 文字條件：把 tokenized 的 caption 丟進（凍結的）text encoder，取得文字 embedding
            encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0]
            if self.args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                self.noise_scheduler.register_to_config(prediction_type=self.args.prediction_type)

            # 訓練目標依 scheduler 設定而不同：
            # - epsilon：目標是「這一步加的雜訊本身」
            # - v_prediction：目標是雜訊與原始 latent 的某種線性組合（速度預測）
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            # UNet 前向傳播：輸入「加噪 latent + 時間步 + 文字 embedding」，輸出預測值（雜訊或 v）
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        with xp.Trace("model.backward"):
            if self.args.snr_gamma is None:
                # 最基本的 loss：預測值跟目標值的均方誤差（MSE）
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                # （進階選項）Min-SNR loss 加權：依訊噪比調整每個時間步的 loss 權重，
                # 讓訓練在不同雜訊強度下更穩定，細節可參考論文連結。
                snr = compute_snr(self.noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
            # 反向傳播算梯度。注意：這裡跟 GPU 一樣是標準的 loss.backward()，
            # 但因為前面是延遲執行，這行呼叫只是把「反向傳播」這件事加進計算圖，
            # 真正的數值運算要等 mark_step() / 取用數值時才會被觸發。
            loss.backward()
        with xp.Trace("optimizer_step"):
            # 更新 UNet 參數（optimizer.step()）。多裝置間的梯度同步不需要手動處理
            # （不像 GPU DDP 需要 all-reduce 梯度），因為 SPMD 底下每個裝置本來就是
            # 對同一份「切分過的」計算圖各自算自己那一塊，XLA 編譯器會在需要的地方自動插入通訊。
            self.run_optimizer()
        return loss


@dataclass
class TrainArgs:
    """訓練腳本的所有命令列參數。

    改用 dataclass + transformers 的 HfArgumentParser 取代原本一長串的
    argparse.add_argument：命令列的用法完全不變（一樣是 --resolution 512、
    --push_to_hub 這種寫法），差別只在於這裡改用「型別註記（type hint）」描述
    每個參數該是什麼型別，程式碼精簡很多，也多了型別檢查跟 IDE 自動補全。

    寫法規則：沒有給 default 值的欄位＝必填參數（對應原本 required=True），
    而且依照 Python dataclass 的規定，這種「沒有 default」的欄位必須排在
    所有「有 default」的欄位最前面。
    """

    pretrained_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )

    profile_duration: int = field(default=10000, metadata={"help": "Profile duration in ms"})
    revision: Optional[str] = field(
        default=None, metadata={"help": "Revision of pretrained model identifier from huggingface.co/models."}
    )
    variant: Optional[str] = field(
        default=None,
        metadata={
            "help": "Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16"
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
                " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
                " or to a folder containing files that 🤗 Datasets can understand."
            )
        },
    )
    train_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "A folder containing the training data. Folder contents must follow the structure described in"
                " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
                " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
            )
        },
    )
    image_column: str = field(default="image", metadata={"help": "The column of the dataset containing an image."})
    caption_column: str = field(
        default="text", metadata={"help": "The column of the dataset containing a caption or a list of captions."}
    )
    output_dir: str = field(
        default="sd-model-finetuned",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory where the downloaded models and datasets will be stored."}
    )
    resolution: int = field(
        default=512,
        metadata={
            "help": (
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            )
        },
    )
    center_crop: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
                " cropped. The images will be resized to the resolution first before cropping."
            )
        },
    )
    random_flip: bool = field(default=False, metadata={"help": "whether to randomly flip images horizontally"})
    train_batch_size: int = field(
        default=16, metadata={"help": "Batch size (per device) for the training dataloader."}
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of training steps to perform.  If provided, overrides num_train_epochs."},
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    snr_gamma: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
                "More details here: https://huggingface.co/papers/2303.09556."
            )
        },
    )
    non_ema_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
                " remote repository specified with --pretrained_model_name_or_path."
            )
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )
    # TPU/XLA 特有：MpDeviceLoader 內部用來預先在 CPU 端準備好幾批資料的佇列大小
    loader_prefetch_size: int = field(
        default=1, metadata={"help": "Number of subprocesses to use for data loading to cpu."}
    )
    loader_prefetch_factor: int = field(
        default=2, metadata={"help": "Number of batches loaded in advance by each worker."}
    )
    # TPU/XLA 特有：控制要提前把幾批資料從 CPU 搬到 TPU 裝置上等著，
    # 目的是讓「資料搬運」跟「TPU 運算」重疊，不要讓 TPU 等資料。這是
    # GPU 訓練裡對應 DataLoader(pin_memory=True) + 手動 prefetch 的 TPU 版本，
    # 但因為 TPU 和 CPU host 是分開的機器（尤其多主機時），這個搬運的角色更重要。
    device_prefetch_size: int = field(
        default=1, metadata={"help": "Number of subprocesses to use for data loading to tpu from cpu. "}
    )
    # 從第幾步開始做效能量測（跳過前面幾步是因為前幾步通常還在做 XLA 圖編譯，會嚴重拉高平均值）
    measure_start_step: int = field(default=10, metadata={"help": "Step to start profiling."})
    adam_beta1: float = field(default=0.9, metadata={"help": "The beta1 parameter for the Adam optimizer."})
    adam_beta2: float = field(default=0.999, metadata={"help": "The beta2 parameter for the Adam optimizer."})
    adam_weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay to use."})
    adam_epsilon: float = field(default=1e-08, metadata={"help": "Epsilon value for the Adam optimizer"})
    prediction_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen."
        },
    )
    # 型別直接寫 Literal["no", "bf16"]，HfArgumentParser 會自動把它變成
    # argparse 的 choices=["no", "bf16"]，效果跟原本 choices=[...] 一樣
    mixed_precision: Optional[Literal["no", "bf16"]] = field(
        default=None, metadata={"help": "Whether to use mixed precision. Bf16 requires PyTorch >= 1.10"}
    )
    push_to_hub: bool = field(default=False, metadata={"help": "Whether or not to push the model to the Hub."})
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    # 開啟後每一步都會印 loss，但會打斷 XLA 延遲執行的批次優化、拉慢訓練速度
    # （原因見 start_training() 裡對 xm.add_step_closure 的說明），預設關閉。
    print_loss: bool = field(default=False, metadata={"help": "Print loss at every step."})

    def __post_init__(self):
        # default to using the same revision for the non-ema model if not specified
        if self.non_ema_revision is None:
            self.non_ema_revision = self.revision


def parse_args():
    # HfArgumentParser 會依照 TrainArgs 裡每個欄位的型別，自動生出對應的
    # argparse 參數（例如 bool 欄位變成 --xxx 開關、Optional[str] 允許不填），
    # parse_args_into_dataclasses() 讀完命令列後直接組成一個 TrainArgs 實例回傳，
    # 用法（args.resolution、args.push_to_hub 等）跟原本的 argparse.Namespace 完全相同。
    parser = HfArgumentParser(TrainArgs)
    (args,) = parser.parse_args_into_dataclasses()
    return args


def setup_optimizer(unet, args):
    optimizer_cls = torch.optim.AdamW
    return optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        foreach=True,
    )


def load_dataset(args):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = datasets.load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    return dataset


def get_column_names(dataset, args):
    column_names = dataset["train"].column_names

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    return image_column, caption_column


def main(args):
    args = parse_args()

    # TPU/XLA 特有：啟動 profiler server，之後 xp.trace_detached() 才能連上來抓 trace
    _ = xp.start_server(PORT)

    # TPU/XLA 特有：查詢目前總共有幾個 TPU 裝置（例如 v5p-128 大約有幾十顆裝置，
    # 視拓樸而定），並建立一個「1 維 mesh」，把所有裝置排成一維、命名為 "data" 這個軸。
    # 之後就是沿著這個 "data" 軸把輸入 batch 切分到各裝置，達成資料平行（data parallel）。
    # 這一段是 GSPMD 的核心設定，GPU 版訓練不會有這種「先描述裝置拓樸網格」的步驟，
    # 通常直接用 torch.distributed.init_process_group() 搭配每個 process 對應一張 GPU。
    num_devices = xr.global_runtime_device_count()
    if xm.is_master_ordinal():
        print(f"num_devices = {num_devices}")
    mesh = xs.get_1d_mesh("data")
    xs.set_global_mesh(mesh)

    # 載入預訓練模型的三大元件：text encoder（CLIP）、VAE、UNet
    # 這步驟目前都還在 CPU 上，之後才會整批搬到 TPU 裝置
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )

    if xm.is_master_ordinal() and args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # TPU/XLA 特有：把 UNet 內部的 nn.Linear 替換成針對 XLA 優化過的版本，
    # 並開啟 XLA 專屬的 flash attention 實作（partition_spec 指定了 attention 運算
    # 內部張量要沿著哪個軸切分到裝置上）。這些是 TPU 執行效率相關的優化補丁，
    # 在 GPU 上通常會改用 xFormers 或 PyTorch 原生的 scaled_dot_product_attention，做法不同。
    from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear

    unet = apply_xla_patch_to_nn_linear(unet, xs.xla_patched_nn_linear_forward)
    unet.enable_xla_flash_attention(partition_spec=("data", None, None, None))

    # VAE 跟 text_encoder 全程凍結（不計算/更新梯度），只有 UNet 會被訓練
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # For mixed precision training we cast all non-trainable weights (vae,
    # non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full
    # precision is not required.
    # 混合精度設定：若指定 bf16，權重會轉成 bfloat16。
    # TPU/XLA 差異：TPU 硬體原生支援 bf16 運算，且 bf16 的數值範圍跟 fp32 幾乎一樣（只是精度較低），
    # 所以這裡不需要像 GPU 上用 fp16 混合精度訓練時那樣，額外搭配
    # torch.cuda.amp.GradScaler 做 loss scaling 來避免數值溢位/下溢。這是 TPU 訓練程式碼
    # 通常比 GPU fp16 版本簡單的原因之一。
    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 取得 XLA 裝置控制代碼。對照 GPU 版通常寫的 torch.device("cuda")，
    # 這裡呼叫的是 xm.xla_device()，之後所有 .to(device) 都是把 tensor/模型搬到 TPU 上。
    device = xm.xla_device()

    # Move text_encode and vae to device and cast to weight_dtype
    # 把三個模型都搬到 TPU 裝置上，並轉成前面決定好的精度
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=weight_dtype)
    optimizer = setup_optimizer(unet, args)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    dataset = load_dataset(args)
    image_column, caption_column = get_column_names(dataset, args)

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)),
            (transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset["train"]
    train_dataset.set_format("torch")
    train_dataset.set_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).to(weight_dtype)
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    g = torch.Generator()
    # 用 host_index()（目前這台 CPU host 是第幾台）當隨機種子的一部分。
    # TPU/XLA 多主機差異：一個大型 TPU pod 通常由多台獨立的 CPU 主機（host）組成，
    # 每台各自跑一份這支腳本、各自讀資料。如果每台的隨機種子一樣，可能會抽到重複的樣本組合，
    # 所以用 host_index 讓每台主機的抽樣序列不同。GPU 多節點訓練也有類似考量，
    # 通常用 torch.distributed 的 rank 來做同樣的事。
    g.manual_seed(xr.host_index())
    # 用「取樣次數趨近無限大」的 RandomSampler，讓 dataloader 可以一直供資料，不會在一個 epoch 後停止；
    # 訓練步數完全由 --max_train_steps 決定，而不是靠「跑完幾個 epoch」。
    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10), generator=g)
    # 這是普通的 PyTorch DataLoader，運作在 CPU 上（負責讀圖片、做 transform、組成 batch），
    # 到這裡為止跟 GPU 訓練寫法沒有差異。
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        prefetch_factor=args.loader_prefetch_factor,
    )

    # TPU/XLA 特有：用 MpDeviceLoader 把上面的 CPU DataLoader「包」起來。
    # 它會在背景把 CPU 準備好的每個 batch 非同步搬到 TPU 裝置上（搬運與 TPU 運算重疊，
    # 避免 TPU 空等資料），對應 GPU 訓練裡「用另一個 CUDA stream 提前把資料搬到 GPU」的角色，
    # 但這裡額外多了 input_sharding 參數：
    #   - "pixel_values": xs.ShardingSpec(mesh, ("data", None, None, None), ...)
    #     表示圖片 tensor 的形狀是 (batch, channel, height, width)，只有第一個維度
    #     （batch 維度）要沿著前面定義的 "data" 軸切分到各個 TPU 裝置，其餘維度不切。
    #   - "input_ids" 同理，只切 batch 維度。
    # 這就是 GSPMD 資料平行的具體實作：每個裝置只拿到整個 batch 裡的一部分資料去跑同一份模型，
    # 而不是像 GPU DDP 那樣，每個 process 各自維護獨立的一份完整 batch、模型副本再手動同步梯度。
    train_dataloader = pl.MpDeviceLoader(
        train_dataloader,
        device,
        input_sharding={
            "pixel_values": xs.ShardingSpec(mesh, ("data", None, None, None), minibatch=True),
            "input_ids": xs.ShardingSpec(mesh, ("data", None), minibatch=True),
        },
        loader_prefetch_size=args.loader_prefetch_size,
        device_prefetch_size=args.device_prefetch_size,
    )

    # TPU/XLA 多主機差異：process_count() 是目前總共有幾台 CPU host 一起參與訓練
    # （單台機器接單一顆或多顆 TPU chip 時通常是 1；大型 pod 會有多台）。
    # 用「總裝置數 / 主機數」算出「平均每台主機分到幾個 TPU 裝置」，純粹是為了下面印統計資訊用。
    num_hosts = xr.process_count()
    num_devices_per_host = num_devices // num_hosts
    # TPU/XLA 多主機差異：xm.is_master_ordinal() 判斷「目前這個行程是不是主節點」。
    # 因為所有主機都在跑同一份程式（SPMD），如果不加這個判斷，
    # 每台主機都會各自印一次同樣的訓練資訊，訊息會重複好幾份。
    if xm.is_master_ordinal():
        print("***** Running training *****")
        print(f"Instantaneous batch size per device = {args.train_batch_size // num_devices_per_host}")
        print(
            f"Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * num_hosts}"
        )
        print(f"  Total optimization steps = {args.max_train_steps}")

    trainer = TrainSD(
        vae=vae,
        weight_dtype=weight_dtype,
        device=device,
        noise_scheduler=noise_scheduler,
        unet=unet,
        optimizer=optimizer,
        text_encoder=text_encoder,
        dataloader=train_dataloader,
        args=args,
    )

    trainer.start_training()
    # 訓練結束後把模型從 TPU 裝置搬回 CPU，才能用一般的 huggingface 方式儲存/上傳
    unet = trainer.unet.to("cpu")
    vae = trainer.vae.to("cpu")
    text_encoder = trainer.text_encoder.to("cpu")

    # 用訓練完的 UNet 組回一個完整的 StableDiffusionPipeline 並存檔，
    # 這樣輸出的資料夾之後可以直接用 StableDiffusionPipeline.from_pretrained() 載入做推理
    # （對照 README「Run inference using the output model」那段範例程式）
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.save_pretrained(args.output_dir)

    if xm.is_master_ordinal() and args.push_to_hub:
        save_model_card(args, repo_id, repo_folder=args.output_dir)
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)