# Python 多进程 (Python Multiprocessing)

## 调试 (Debugging)

有关已知问题以及如何解决它们的信息，请参阅 [问题排查](../usage/troubleshooting.md#python-multiprocessing) 页面。

## 简介 (Introduction)

!!! important "重要"
    本文档中对源码的引用，代表的是 2024 年 12 月撰写本文档时的代码状态。

vLLM 中 Python 多进程的使用之所以复杂，主要原因在于：

- 将 vLLM 作为库使用时，对其内部代码的控制受到限制；
- 某些多进程启动方法与 vLLM 的依赖库之间存在不兼容性。

本文档描述了 vLLM 是如何应对这些挑战的。

## 多进程启动方法 (Multiprocessing Methods)

[Python 多进程启动方法](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods) 包括：

- `spawn`：启动一个新的 Python 进程。Windows 和 macOS 上的默认方式。
- `fork`：使用 `os.fork()` 分叉 Python 解释器。对于 Python 3.14 之前的版本，Linux 上的默认方式。
- `forkserver`：启动一个服务器进程，该服务器进程会根据请求分叉出新的进程。对于 Python 3.14 及更高版本，Linux 上的默认方式。

### 权衡 (Tradeoffs)

- `fork` 是最快的方法，但与使用线程的依赖项不兼容。如果您在 macOS 下，使用 `fork` 可能会导致进程崩溃。
- `spawn` 与依赖项的兼容性更好，但在将 vLLM 作为库使用时可能会出现问题。如果调用方代码未使用 `__main__` 保护（`if __name__ == "__main__":`），当 vLLM 启动新进程时，调用方的代码会被无意中重新执行。这可能会导致无限递归以及其他问题。
- `forkserver` 会启动一个新的服务器进程，该进程会根据需要分叉出新进程。不幸的是，当 vLLM 作为库使用时，它面临与 `spawn` 相同的问题。服务器进程是作为一个新启动的进程创建的，它将重新执行未受 `__main__` 保护的代码。

对于 `spawn` 和 `forkserver`，进程都绝不能像 `fork` 那样依赖于继承任何全局状态。

## 与依赖项的兼容性

多个 vLLM 的依赖项表明它们要么更倾向于使用 `spawn`，要么将其作为硬性要求：

- <https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing>
- <https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors>
- <https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders>

已知在初始化这些依赖项之后使用 `fork` 会出现问题。

## 当前状态 - v0 (Current State - v0)

环境变量 `VLLM_WORKER_MULTIPROC_METHOD` 可用于控制 vLLM 所使用的多进程方法。当前的默认值是 `fork`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L339-L342>

如果主进程是通过 `vllm` 命令行命令控制的，则会使用 `spawn`，因为它的兼容性最广。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/scripts.py#L123-L140>

`multiproc_xpu_executor` 会强制使用 `spawn`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/executor/multiproc_xpu_executor.py#L14-L18>

还有其他一些地方硬编码使用了 `spawn`：

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/distributed/device_communicators/all_reduce_utils.py#L135>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/entrypoints/openai/api_server.py#L184>

相关的 PR：

- <https://github.com/vllm-project/vllm/pull/8823>

## 曾经在 v1 中的状态 (Prior State in v1)

曾经有一个环境变量来控制是否在 v1 引擎核心中启用多进程，即 `VLLM_ENABLE_V1_MULTIPROCESSING`。该变量默认关闭。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454>

当启用它时，v1 的 `LLMEngine` 会创建一个新进程来运行引擎核心。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45>

由于前面提到的所有原因（即与依赖项的兼容性，以及将 vLLM 作为库使用的代码兼容性），它在默认情况下是关闭的。

### v1 中所做的更改

使用 Python 的 `multiprocessing` 并没有一个能够完美适用于任何地方的简单解决方案。作为第一步，我们可以将 v1 调整到一种状态：采取“尽力而为（best effort）”的策略来选择多进程方法，以实现兼容性最大化。

- 默认使用 `fork`。
- 当我们明确知道控制主进程时（即执行了 `vllm` 命令行命令），使用 `spawn`。
- 如果我们检测到 `cuda` 此前已被初始化，则强制使用 `spawn` 并发出警告。因为我们知道这种情况下使用 `fork` 会发生崩溃，所以这是我们目前能做的最好选择。

在此场景下，目前已知仍会崩溃的情况是：在调用 vLLM 之前初始化了 `cuda` 的库调用代码。我们发出的警告信息应当指示用户要么添加一个 `__main__` 保护，要么禁用多进程。

如果发生这种已知失效的情况，用户将会看到两条解释原因的消息。首先是来自 vLLM 的日志警告信息：

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

其次，Python 自身会抛出一个带有详尽解释的异常：

```console
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

## 考虑过的备选方案

### 检测是否存在 `__main__` 保护

有人建议，如果我们能检测将 vLLM 作为库使用的代码中是否含有 `__main__` 保护，那么我们就能采取更好的行为。[Stack Overflow 上的这篇帖子](https://stackoverflow.com/questions/77220442/multiprocessing-pool-in-a-python-class-without-name-main-guard) 就是来自一位面临相同问题的库作者的讨论。

我们确实可以检测我们是在最初的 `__main__` 进程中，还是在随后的派生进程中。然而，要在代码中直接检测是否存在 `__main__` 保护似乎并不简单。

因此，该选项因不切实际而被放弃。

### 使用 `forkserver`

起初，`forkserver` 看起来是一个很好的解决方案。然而，它的工作方式在将 vLLM 作为库使用时，面临与 `spawn` 相同的挑战。

### 始终强制使用 `spawn`

整理此事的一种方式是始终强制使用 `spawn`，并在文档中写明，在将 vLLM 作为库使用时必须加上 `__main__` 保护。但这不幸地会破坏现有的代码，并使 vLLM 的使用变得更加困难，违背了使 `LLM` 类尽可能易用的初衷。

因此，我们没有将这种复杂性推给我们的用户，而是选择在 vLLM 内部保留这些复杂性，以尽力使一切正常运行。

## 未来的工作

未来我们可能想要考虑一种不同的 Worker 管理方法，以规避这些挑战。

1. 我们可以实现类似 `forkserver` 的东西，但让进程管理器成为我们最初通过运行自己的子进程和专用于 Worker 管理的自定义 entrypoint 来启动的内容（例如启动一个 `vllm-manager` 进程）。
2. 我们可以探索其他可能更适合我们需求的库。例如可以考虑：
   - <https://github.com/joblib/loky>
