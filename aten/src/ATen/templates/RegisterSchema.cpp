// ${generated_comment}
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ops/empty.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace at {
TORCH_LIBRARY(aten, m) {
  ${aten_schema_registrations};
  // Distributed Ops
  // Implementations located in torch/csrc/jit/runtime/register_distributed_ops.cpp
  m.def("get_gradients(int context_id) -> Dict(Tensor, Tensor)");
  // https://zhuanlan.zhihu.com/p/681408472 在 C++中注册一个分发的运算符
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}

Tensor myadd_cpu(const Tensor& self_, const Tensor& other_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(other_.device().type() == DeviceType::CPU);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = empty(self.sizes(), self.options());
  const float* self_ptr = self.data_ptr<float>();
  const float* other_ptr = other.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("myadd", myadd_cpu);
}

/**
 * 这个函数是算子的所有key共用的，例如autograd / cpu / gpu / ...，所有这些dispatcher key，当它们
 * 运行完后，都会调用这个函数，把任务交给下一个dispatcher key。例如算子总共有autograd cpu两个dispatcher key，
 * 则autograd运行完后，调用这个函数就会交给cpu运算
*/
Tensor myadd(const Tensor& self, const Tensor& other) {
  /**
   * 这里的op是static的，说明该算子所有任务都共享同一个op，所以这个op应该就是算子的无状态的dispatch key sets，
   * 根据https://zhuanlan.zhihu.com/p/376495783，op应该就是这个bit set，不同线程的任务之所以知道下一个key，
   * 是因为它们在线程变量里维护自己的local exclude，当运行完一个key后就mask掉，避免下次再运行同个key。
   * 因为op整个进程周期只初始化一次，所以耗时可以忽略。
  */
  static auto op = Dispatcher::singleton()
                       .findSchemaOrThrow("aten::myadd", "")
                       .typed<decltype(myadd)>();
  return op.call(self, other);
}

class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
 public:
  /**
   * 先调度到autograd的forward，再通过‘at::AutoNonVariableTypeMode g’
   * 把autograd放到local exclude（不然会递归一直进这里）
   * 最后调用`myadd`调度函数，交给下个key处理（下个key不一定是cpu/gpu）
   */
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      Tensor self,
      Tensor other) {
    at::AutoNonVariableTypeMode g;
    return myadd(self, other);
  }

  // add操作的梯度，就是把父节点的梯度，直接给两个子节点
  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    return {grad_outputs[0], grad_outputs[0]};
  }
};

Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
  return MyAddFunction::apply(self, other)[0];
}

TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("myadd", myadd_autograd);
}

${schema_registrations}
}  // namespace at
