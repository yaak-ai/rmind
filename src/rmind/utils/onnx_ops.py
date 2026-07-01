import torch
from onnxscript.function_libs.torch_lib import _constants as _onnxscript_constants
from onnxscript.onnx_opset import opset18 as op
from onnxscript.values import Opset, TracedOnnxFunction


def _aten_gru_linear_before_reset(
    input,
    hx,
    params,
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
):
    """Corrected aten.gru.input translation that sets linear_before_reset=1.

    The onnxscript torchlib default (aten_gru in ops/core.py) omits
    linear_before_reset, so ONNX Runtime defaults to 0. PyTorch's GRU
    applies the hidden linear transform before the reset gate multiplication
    (i.e. linear_before_reset=1), causing ~10% relative error without this fix.
    """
    num_directions = 2 if bidirectional else 1

    if batch_first:
        input = op.Transpose(input, perm=[1, 0, 2])

    hidden_size = op.Shape(hx, start=2, end=3)
    current_input = input
    output_h_list = []

    for layer_idx in range(num_layers):
        layer_h = op.Slice(hx, [layer_idx * num_directions], [(layer_idx + 1) * num_directions], axes=[0])

        params_per_direction = 4 if has_biases else 2
        param_start_idx = layer_idx * params_per_direction * num_directions

        W_list, R_list = [], []
        B_list = [] if has_biases else None

        for dir_idx in range(num_directions):
            p = param_start_idx + dir_idx * params_per_direction
            W_ih = params[p]      # [3*H, input_size], PyTorch order: r,z,n
            W_hh = params[p + 1]  # [3*H, H],          PyTorch order: r,z,n

            # Reorder gates: PyTorch [r,z,n] → ONNX [z,r,n]
            W_ir = op.Slice(W_ih, starts=[0],            ends=hidden_size,      axes=[0])
            W_iz = op.Slice(W_ih, starts=hidden_size,    ends=hidden_size * 2,  axes=[0])
            W_in = op.Slice(W_ih, starts=hidden_size * 2, ends=hidden_size * 3, axes=[0])
            W_hr = op.Slice(W_hh, starts=[0],            ends=hidden_size,      axes=[0])
            W_hz = op.Slice(W_hh, starts=hidden_size,    ends=hidden_size * 2,  axes=[0])
            W_hn = op.Slice(W_hh, starts=hidden_size * 2, ends=hidden_size * 3, axes=[0])

            W_list.append(op.Unsqueeze(op.Concat(W_iz, W_ir, W_in, axis=0), [0]))
            R_list.append(op.Unsqueeze(op.Concat(W_hz, W_hr, W_hn, axis=0), [0]))

            if has_biases:
                b_ih = params[p + 2]  # [3*H], PyTorch order: r,z,n
                b_hh = params[p + 3]  # [3*H], PyTorch order: r,z,n

                b_ir = op.Slice(b_ih, starts=[0],            ends=hidden_size,      axes=[0])
                b_iz = op.Slice(b_ih, starts=hidden_size,    ends=hidden_size * 2,  axes=[0])
                b_in = op.Slice(b_ih, starts=hidden_size * 2, ends=hidden_size * 3, axes=[0])
                b_hr = op.Slice(b_hh, starts=[0],            ends=hidden_size,      axes=[0])
                b_hz = op.Slice(b_hh, starts=hidden_size,    ends=hidden_size * 2,  axes=[0])
                b_hn = op.Slice(b_hh, starts=hidden_size * 2, ends=hidden_size * 3, axes=[0])

                B_list.append(op.Unsqueeze(
                    op.Concat(b_iz, b_ir, b_in, b_hz, b_hr, b_hn, axis=0), [0]  # [1, 6*H]
                ))

        W = op.Concat(*W_list, axis=0) if len(W_list) > 1 else W_list[0]
        R = op.Concat(*R_list, axis=0) if len(R_list) > 1 else R_list[0]
        direction = "bidirectional" if bidirectional else "forward"
        hidden_size_attr = hx.shape[2]

        if has_biases:
            B = op.Concat(*B_list, axis=0) if len(B_list) > 1 else B_list[0]
            Y, Y_h = op.GRU(
                current_input, W, R, B,
                initial_h=layer_h,
                direction=direction,
                hidden_size=hidden_size_attr,
                linear_before_reset=1,
            )
        else:
            Y, Y_h = op.GRU(
                current_input, W, R,
                initial_h=layer_h,
                direction=direction,
                hidden_size=hidden_size_attr,
                linear_before_reset=1,
            )

        # Y: [seq, num_directions, batch, H] → [seq, batch, num_directions*H]
        Y = op.Transpose(Y, perm=[0, 2, 1, 3])
        Y_shape = op.Shape(Y)
        new_shape = op.Concat(
            op.Slice(Y_shape, [0], [1]),
            op.Slice(Y_shape, [1], [2]),
            op.Reshape(
                op.Mul(op.Slice(Y_shape, [2], [3]), op.Slice(Y_shape, [3], [4])),
                op.Constant(value_ints=[-1]),
            ),
            axis=0,
        )
        current_input = op.Reshape(Y, new_shape)

        if layer_idx < num_layers - 1 and dropout > 0.0 and train:
            current_input, _ = op.Dropout(current_input, dropout, train)

        output_h_list.append(Y_h)

    final_h = output_h_list[0] if len(output_h_list) == 1 else op.Concat(*output_h_list, axis=0)

    if batch_first:
        current_input = op.Transpose(current_input, perm=[1, 0, 2])

    return current_input, final_h


GRU_CUSTOM_TABLE = {
    torch.ops.aten.gru.input: TracedOnnxFunction(
        Opset(domain=_onnxscript_constants.DOMAIN, version=1),
        _aten_gru_linear_before_reset,
    )
}
