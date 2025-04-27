# deepspeed介绍
https://github.com/microsoft/DeepSpeed
https://arxiv.org/abs/1910.02054

## 1. 为什么需要DeepSpeed 

- 分布式计算环境中，主节点负责协调其他节点和进程的工作
- 混合精度训练
ZeRO可以减少内存占用，优化大模型训练，将模型参数分成了三个部分：Optimizer States、Gradient 和 Model Parameter。在使用 ZeRO 进行分布式训练时，可以选择 ZeRO-Offload 和 ZeRO-Stage3 等不同的优化技术。

### 简介
DeepSpeed是一个由微软开发的开源深度学习优化库，旨在**提高大规模模型训练的效率和可扩展性**。它通过多种技术手段来加速训练，包括模型并行化、梯度累积、动态精度缩放、本地模式混合精度等。DeepSpeed还提供了一些辅助工具，如分布式训练管理、内存优化和模型压缩等，以帮助开发者更好地管理和优化大规模深度学习训练任务。此外，deepspeed基于pytorch构建，只需要简单修改即可迁移。DeepSpeed已经在许多大规模深度学习项目中得到了应用，包括语言模型、图像分类、目标检测等等。

### 软件架构
1. Apis：提供易用的api接口，训练模型、推理模型只需要简单调用几个接口即可。其中最重要的是initialize接口，用来初始化引擎，参数中配置训练参数及优化技术等。配置参数一般保存在config.json文件中。
2. runtime：运行时组件，是deepspeed管理、执行和性能优化的核心组件。如部署训练任务到分布式设备、数据分区、模型分区、系统优化、微调、故障检测、checkpoints保存和加载等。该组件使用python语言实现。
3. ops：用c++和cuda实现底层内核，优化计算和通信，例如ultrafast transformer kernels, fuse LAN kernels, customary deals等。


## **2. Zero(Zero Redundancy Optimizer)**

微软开发ZeRO是为了克服数据并行性和模型并行性的限制，同时实现两者的优点。**ZeRO通过在数据并行进程中划分模型状态（参数，梯度和优化器状态），而不是复制它们，从而消除了数据并行进程中的内存冗余。它在训练期间使用动态通信计划，以在分布式设备之间共享必要的状态，以保持计算粒度和数据并行性的通信量**\*\*。\*\*&#x20;

ZeRO驱动的数据并行性，它允许每个设备的内存使用量随数据并行性的程度线性扩展，并产生与数据并行性相似的通信量。 ZeRO支持的数据并行性可以适合任意大小的模型，只要**聚合的设备内存足够大以共享模型状态**即可。

ZeRO是一种用于大规模训练优化的技术，主要是用来减少内存占用。在大规模训练中，内存占用可以分为 Model States 和 Activation 两部分，而 ZeRO 主要是为了解决 Model States 的内存占用问题。

ZeRO 将模型参数分成了三个部分：Optimizer States、Gradient 和 Model Parameter。

-   `Optimizer States` 是 Optimizer 在进行梯度更新时所需要用到的数据，例如 SGD 中的 Momentum。
-   `Gradient `是在反向传播后所产生的梯度信息，其决定了参数的更新方向。
-   `Model Parameter` 则是模型参数，也就是我们在整个过程中通过数据“学习”的信息。


### **2.1 三个级别**

`ZeRO-0`：禁用所有类型的分片，仅使用 DeepSpeed 作为 DDP (Distributed Data Parallel)

`ZeRO-1`：分割Optimizer States，减少了4倍的内存，通信容量与数据并行性相同

`ZeRO-2`：分割Optimizer States与Gradients，8x内存减少，通信容量与数据并行性相同

`ZeRO-3`：分割Optimizer States、Gradients与Parameters，内存减少与数据并行度和复杂度成线性关系。

`ZeRO-Infinity`是ZeRO-3的拓展。允许通过使用 NVMe 固态硬盘扩展 GPU 和 CPU 内存来训练大型模型。ZeRO-Infinity 需要启用 ZeRO-3。

在deepspeed中通过zero\_optimization.stage=0/1/2/3 设置，

卸载通过zero\_optimization.offload\_ optimizer.device设置

### **2.2 混合精度**

混合精度训练是指在训练过程中同时使用FP16（半精度浮点数）和FP32（单精度浮点数）两种精度的技术。**使用FP16可以大大减少内存占用，从而可以训练更大规模的模型**。但是，**由于FP16的精度较低，训练过程中可能会出现梯度消失和模型不稳定的问题**。因此，需要使用一些技术来解决这些问题，例如\*\*动态精度缩放（Dynamic Loss Scaling）**和**混合精度优化器（Mixed Precision Optimizer）\*\*等。

![](image/image_6GC207ZU3O.png)

deepspeed提供了混合精度训练的支持，可以通过在配置文件中设置`"fp16.enabled": true`来启用混合精度训练。在训练过程中，deepspeed会自动将一部分操作转换为FP16格式，并根据需要动态调整精度缩放因子，从而保证训练的稳定性和精度。

在使用混合精度训练时，需要注意一些问题，例如梯度裁剪（Gradient Clipping）和学习率调整（Learning Rate Schedule）等。梯度裁剪可以防止梯度爆炸，学习率调整可以帮助模型更好地收敛。因此，在设置混合精度训练时，需要根据具体情况进行选择和配置。

![](image/image_vfi8OaGD7t.png)

**BF16**

**BF16和FP16都是混合精度训练中使用的浮点数表示格式**。

![](image/image_l1dkF_7Tg7.png)

BF16是一种Brain Floating Point格式，可以提供更好的数值稳定性和更高的精度，使用的存储空间和`float16`相同。在混合精度训练中，**BF16可以作为一种精度更高的替代品，用于一些关键的计算操作，例如梯度累加和权重更新等**。使用BF16可以提高模型的训练速度和精度，并减少内存占用。

在 DeepSpeed 中，可以通过在配置文件中设置 `"bf16.enabled": true` 来启用 BF16 混合精度训练。这将会将一部分操作转换为 BF16 格式，并根据需要动态调整精度缩放因子，从而提高模型的训练速度和精度，并减少内存占用。


### **2.3 显存占用分析**

混合精度训练是指存在fp16和fp32两种格式的数值，其中模型参数、模型梯度都是fp16，此外还有fp32的模型参数，如果优化器是Adam，则还有fp32的momentum和variance。

总的来说，模型训练时显存主要分为两部分。

-   **第一部分**是模型权重、梯度和优化器状态；
-   **第二部分**是激活和临时缓存区。

**ZeRO-DP主要是优化第一部分的显存占用，所以这里主要介绍第一部分的显存。**&#x20;

![](image/image_XNjij0Z1Dh.png)

![](image/image_B3Zt993aVo.png)

-   **将权重转换为FP16**：在这一步中，神经网络的权重（或参数）最初是FP32格式，被转换为低精度的FP16格式。这减少了内存的占用，并允许更快的计算，因为FP16操作需要更少的内存，并且可以被硬件更快地处理。 &#x20;
-   **计算梯度**：神经网络的前向和后向传播是使用较低精度的FP16权重进行的。
-   **将梯度转换为FP32**：在FP16中计算梯度后，它们被转换回高精度的FP32格式。这种转换对于保持数值稳定性和避免使用低精度算术时可能出现的梯度消失或爆炸等问题至关重要。 &#x20;
-   **乘以学习率和更新权重**：现在是FP32格式，梯度被乘以学习率（一个标量值，决定了优化过程中的步长）。乘积被用来更新原始FP32神经网络权重。学习率有助于控制优化过程的收敛性，对于实现良好的性能至关重要。

#### （1）**模型状态**（model states）

假设模型的参数量是 $Ψ$ ，使用Adam为优化器进行混合精度训练。

1.  由于模型的参数和梯度使用float16，所以显存消耗分别为 $2Ψ$ 和 $2Ψ$ 。
2.  Adam会维护一个float32的模型备份副本，消耗 $4Ψ$ 显存。Adam优化器本身会为模型的每个参数维护两个float32的辅助变量（fp32的momentum和fp32的variance），所以显存消耗占用为 $4Ψ+4Ψ$ 。

总的来说，模型会消耗 $2Ψ+2Ψ=4Ψ$ ，Adam优化器这消耗$ 4Ψ+4Ψ+4Ψ=12Ψ $。最终的总消耗为 $4Ψ+12Ψ=16Ψ $。

![](image/image_pcdg2zZLBJ.png)

\*\*优化器显存占用表示为 **$KΨ$** (不同的优化器不同)，混合精度训练的显存占用为 **$4Ψ+KΨ$** 。 \*\*

来看一个例子，**qwen2.5-0.5B**含有0.5B个参数，如果用fp16格式，只需要`0.5G*2Byte=1GB`显存

但是模型状态实际上(使用adam优化器)需要耗费`0.5*16=8GB`, 相比之下，激活值可以用[activation checkpointing](https://arxiv.org/pdf/1604.06174.pdf "activation checkpointing")来大大减少，所以模型状态就成了头号显存杀手，它也是ZeRO的重点优化对象。而其中Adam状态又是第一个要被优化的。

比如说有一个模型参数量是1M，在一般的深度学习框架中(比如说PyTorch)，一般是32位存储。32位存储的意思就是1个参数用32个bit来存储。那么这个拥有1M参数量的模型所需要的存储空间的大小即为：1M \* 32 bit = 32Mb = 4MB。因为1 Byte = 8 bit。现在的quantization技术就是减少参数量所占的位数：比如我用16位存储，那么：所需要的存储空间的大小即为：1M \* 16 bit = 16Mb = 2MB。


## **3.ZeRO-DP**

ZeRO-DP(Zero Redundancy Optimizer-Data Parallelism)是来自于论文《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》中的一种显存优化方法ZeRO的核心部分。通过该方法可以大幅度的优化显存占用，**从而在有限的资源下训练更大的模型**。

针对模型状态的存储优化（去除冗余），ZeRO使用的方法是分片（partition），即每张卡只存 1/N的模型状态量，这样系统内只维护一份模型状态。

**这里os指的是optimizer**

![](image/image_UKoVaOXc-2.png)

这个概念和一般的DP的差别，只是没有每个 GPU 都复制完整的模型参数、梯度和优化器状态，而是每个 GPU 只存储其中的一部分。在随后的运行过程中，当需要给定层的完整层参数时，所有 GPU 同步以相互提供它们缺失的部分 —— 仅此而已。

第二列给出了一个示例：$  K=12,Ψ=7.5B,N=64  $可以看到显存优化相当明显。

在标准的数据并行中，每个显卡(rank)都会保存独立的**权重、梯度和优化器状态**，如上图中的baseline所示。那么每个显卡是否有必要存储全部的这些信息呢？**ZeRO-DP的答案是不需要**。ZeRO-DP能够对模型状态(权重、梯度和优化器状态)进行划分(不像标准DP那样进行复制)，然后通过动态通信调度来最小化通信开销。ZeRO-DP能够在保持整体通信开销接近标准DP的同时，线性地降低模型的**单显卡**显存占用。

### **3.1 ZeRO-DP的细节**

总的来说，ZeRO-DP可以分为三个阶段：**Pos, Pg, Pp** 。三个阶段对应优化器状态划分、梯度划分和模型参数划分，并且三个阶段可以叠加使用(上图展示了三个阶段的叠加)。关于三个阶段是否会增加通信量，会在后面分析，目前先接受这三个阶段并不会显著增加通信开销。

![](image/image_psVU2KOWUS.png)
- **all-gather**
**通信量： $Ψ$**：
基本定义
All-Gather 操作将分布在不同进程上的数据片段收集起来，并将完整的数据集复制到每个进程上。
工作原理
假设有 N 个进程，每个进程拥有一个数据片段：
每个进程将自己的数据片段发送给所有其他进程
每个进程接收来自所有其他进程的数据片段
每个进程将收到的所有片段按照进程编号顺序组合成完整的数据集

![](image/image_tOHRS0iiUq.png)
- **reduce-scatter** **通信量： $Ψ$**:
基本定义
Reduce-Scatter 操作将每个进程上的数据先进行归约（如求和），然后将结果分散到所有进程上，使每个进程只获得最终结果的一部分。
工作原理
假设有 N 个进程，每个进程都有一个大小为 N 的数组：
Reduce 阶段：对所有进程中相同位置的元素执行归约操作（通常是求和）
Scatter 阶段：将归约结果分散，使第 i 个进程获得归约结果的第 i 个部分

![](image/image_JxB5Yju8gj.png)

在DeepSpeed中，一般使用ZeRO-1就足够了。

![](image/image_hKE_Xt759j.png)

### 3.2 **ZeRO-DP通信量**

ZeRO通过去除显存的冗余来提升模型尺寸，那么该方法是否是通过通信量换取的显存效率。换句话说，ZeRO-DP相较于标准DP来说，通信量增大了吗？

答案分为两部分：

1.  **ZeRO-DP在使用** Pos **和** Pg**的情况下，能够带来8倍的显存降低且不增加额外的通信量；**
2.  **当同时使用** Pos **、** Pg **和**Pp**时，通信量增加1.5倍，同时降低倍的显存。**&#x20;

在分析之前，我们先回顾下常用的集合通信（collective communication）函数[Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html "Collective Operations")。

#### **（1）标准数据并行的通信量**

在标准的数据并行训练中，在反向传播结束后，跨显卡的梯度会被平均。这个平均的过程使用all-reduce。对于大尺寸的模型，all-reduce通信是整个通信带宽的上界，因此分析主要集中在all-reduce上。

传统数据数据并行在每一步（step/iteration）计算梯度后，需要进行一次AllReduce操作来计算梯度均值，目前常用的是Ring AllReduce，分为ReduceScatter和AllGather两步，每张卡的通信数据量（发送+接受）。总的来说，单个显卡在reduce-scatter或者all-gather的过程中，都会有 Ψ 的通信量。那么，整个all-reduce的单显卡通信量为 2Ψ 。

参考：[\[深度学习\]Ring All-reduce的数学性质-CSDN博客](https://zengwenqi.blog.csdn.net/article/details/130501965 "\[深度学习]Ring All-reduce的数学性质-CSDN博客")

#### **（2）Zero-DP的通信量**

Pos**的通信量**

在单独使用 Pos的情况下，单个显卡会保存完整的模型参数和梯度。随后使用reduce-scatter将梯度reduce至不同的显卡上(**此时不同显卡仅拥有完整平均梯度的一部分**)，该步骤的通信量是 Ψ 。各个显卡使用部分梯度更新对应的优化器状态，然后再更新对应的参数(**此时每个显卡上的模型都更新了一部分参数**)。最后，使用all-gather将分布在各个显卡上的更新后参数分发自所有显卡上(**此时所有显卡上都有了完整的更新后参数**)，该步骤的通信量是 Ψ 。总的来说，各个显卡仅需要持有部分优化器状态即可，且总的通信量仍然是 2Ψ 。

## **4.DeepSpeed训练**

### 4.1 基本训练的介绍

安装 DeepSpeed：

```bash
pip install deepspeed

```

1.  在训练脚本中导入 DeepSpeed 模块：
2.  在训练脚本中导入 Trainer 模块：
3.  创建 Trainer 对象，将模型、训练数据集、优化器等参数传入：

```python
import deepspeed
 
from transformers import Trainer
 
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    optimizer=optimizer,
)
trainer.train()
```

1.  使用 DeepSpeed 命令行工具运行训练脚本（单机）：

```bash
deepspeed --num_gpus=4 train.py

```

其中，`--num_gpus` 表示使用的 GPU 数量。

```python

# 定义用于打印显存占用的回调
class MemoryUsageCallback(TrainerCallback):
    def __init__(self, print_every_n_steps=10):
        self.print_every_n_steps = print_every_n_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 只在主进程 (rank 0) 且达到打印步数时打印
        if state.is_world_process_zero and state.global_step % self.print_every_n_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保 CUDA 操作完成
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"\n[Step {state.global_step}] GPU Memory Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def sft_train(args):
    # 禁用 tokenizers 库的并行处理以避免 DDP/fork 死锁警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.cuda.set_device(args.local_rank)
    # 设置随机种子
    if hasattr(args, 'seed'):
        set_seed(args.seed)
    else:
        set_seed(42)  # 默认使用42作为随机种子

    # 设置环境变量以优化内存分配
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 加载模型 - 不使用device_map，依赖LoRA减少内存需求
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        cache_dir=args.cache_dir,
        trust_remote_code=True,  # 添加以支持Qwen等模型
    )

    # 打印模型加载后的显存占用 (仅在 rank 0 执行)
    current_rank = int(os.environ.get("RANK", -1)) # RANK 由 deepspeed launcher 设置
    if current_rank == 0 or current_rank == -1:
        # 确保所有 CUDA 操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Rank {current_rank}] Model loaded. GPU Memory Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        else:
            print(f"[Rank {current_rank}] Model loaded, but CUDA not available.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # 准备数据集
    train_dataset = get_sft_dataset(
        path='train.json', 
        sft=True, 
        cache_dir=args.cache_dir
    )

    ds_config = {
      "train_batch_size": "auto", 
      "train_micro_batch_size_per_gpu": "auto", # 每个GPU上的微批处理大小
      "gradient_accumulation_steps": "auto",

      "optimizer": {
        "type": "AdamW", 
        "params": {
          "lr": "auto" 
        }
      },
      "scheduler": {
        "type": "WarmupLR",
        "params": {
          "warmup_min_lr": "auto", 
          "warmup_max_lr": "auto",   
          "warmup_num_steps": "auto" 
        }
      },

      "fp16": {
        "enabled": False 
      },
      "bf16": {
         "enabled": "auto" 
      },

      "zero_optimization": {
         "stage": 2 # 可以在每个流水线阶段内部使用ZeRO-1/2来优化
      },

      "gradient_clipping": "auto"
    }

    # 准备训练参数
    training_args = SFTConfig(
        output_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        max_grad_norm=args.max_grad_norm,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # 启用梯度检查点
        dataloader_num_workers=4,  # 增加数据加载器的工作进程数
        dataloader_pin_memory=True,  # 启用数据加载器的内存固定
        deepspeed=ds_config,
    )

    # 实例化回调，设置打印频率 (例如每 10 步打印一次)
    memory_callback = MemoryUsageCallback(print_every_n_steps=10)

    # 初始化训练器，传入回调
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[memory_callback]  # <--- 添加回调
    )

    # 开始训练
    trainer.train()
```

```bash
deepspeed --num_gpus=4   main.py --task sft_train --subtask sft  --checkpoint_dir ./model/sft_checkpoint
```

### ZeRO1 $P_{os}$

GPU Memory Allocated: 5.55 GB
| 显存组成部分 | 大小 (估算) | 解释 |
| :----------------------------- | :---------- | :------------------------------------------------------------------- |
| 模型参数 (Model Parameters) | 1.0 GB | ZeRO-1 每个 GPU 保留完整模型副本。 |
| 优化器状态 (Optimizer States) | 1.0 GB | 总共 4 GB (FP32)，分片到 4 个 GPU，每个 GPU 承担 4 GB / 4 = 1 GB。 |
| 梯度 (Gradients) | 1.0 GB | 每个 GPU 在反向传播峰值时需要持有完整的梯度副本，用于后续的 AllReduce。 |
| 激活值 (Activations) | ~2.0 GB | 这是根据总显存反推出来的最主要变量。4K长序列会产生大量激活值。 |
| 框架及通讯缓存 | ~0.5 GB | PyTorch/CUDA 基础开销 + DeepSpeed 用于 AllReduce 的通讯缓存等。 |
| 总计 (估算) | ~5.5 GB | 1.0 + 1.0 + 1.0 + 2.0 + 0.5 = 5.5 GB |
关键点: ZeRO-1 的峰值显存较高，主要是因为它需要在每个 GPU 上临时存储完整的梯度。

### ZeRO1 $P_{os} + P_{g}$

GPU Memory Allocated: 3.74 GB
化器状态和梯度都被分片。
| 显存组成部分 | 大小 (估算) | 解释 |
| :----------------------------- | :---------- | :---------------------------------------------------------------------------- |
| 模型参数 (Model Parameters) | 1.0 GB | ZeRO-2 每个 GPU 仍然保留完整模型副本。 |
| 优化器状态 (Optimizer States) | 1.0 GB | 同 ZeRO-1，总共 4 GB (FP32)，分片到 4 个 GPU，每个 GPU 承担 1 GB。 |
| 梯度 (Gradients) | 0.25 GB | 总共 1 GB (BF16)，分片到 4 个 GPU，每个 GPU 只需负责 1 GB / 4 = 0.25 GB。 |
| 激活值 (Activations) | ~1.0 GB |  |
| 框架及通讯缓存 | ~0.5 GB | PyTorch/CUDA 基础开销 + DeepSpeed 用于 ReduceScatter 的通讯缓存（可能比ZeRO-1略小）。|
| 总计 (估算) | ~3.75 GB| 1.0 + 1.0 + 0.25 + 1.0 + 0.5 = 3.75 GB (与观察到的 3.74 GB 非常接近) |

关键点: ZeRO-2 显著降低了显存占用，因为它：
分片了梯度： 每个 GPU 只需存储 1/4 的梯度，直接节省了 0.75 GB。
降低了峰值需求： 通过 ReduceScatter 操作，不需要在单个 GPU 上完整地聚合梯度，减少了反向传播期间的峰值内存需求和通讯缓存需求，影线了剩余的 ~1 GB 显存下降。