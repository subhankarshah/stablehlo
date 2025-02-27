## About

When bootstrapping StableHLO from MHLO, we have inherited MHLO's implementation
of many things, including prettyprinting, verification and shape inference.
Thanks to that, we already have significant coverage of the opset, but there's
still plenty to do to review the existing implementations for completeness and
provide new implementations where none exist.

This live document is for the developers and the users to track the progress on
various aspects of the opset - specification, verification, type inference,
pretty printing, interpreter, etc.

### How to use it

The progress of a StableHLO op, as mentioned in the corresponding row, on a
particular aspect, as mentioned in the corresponding column, is tracked using
one of the following tracking labels.

 - Generic labels
    - **yes**: there is a comprehensive implementation.
    - **no**: there is no implementation, but working on that is part of
      [the roadmap](https://github.com/openxla/stablehlo#roadmap).
      Note that Verifier can never be labeled as "no" because the ODS already
      implements some verification.
 - Customized labels for Verifier and Type Inference
    - **yes**: there is an implementation, and it's in sync with
      [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md).
    - **yes\***: there is an implementation, and it's in sync with
      [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
      Since XLA semantics is oftentimes underdocumented, we are using
      [hlo_verifier.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo_verifier.cc)
      and [shape_inference.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/shape_inference.cc)
      as the reference.
    - **revisit**: there is an implementation, but it doesn't fall under "yes"
      or "yes\*" - either because we haven't audited it yet, or because we have
      and found issues.
    - **infeasible**: there is no implementation, because it's infeasible.
      For example, because the result type of an op cannot be inferred from
      its operands and attributes.

## Status

| StableHLO Op             | Specification | Verification | Type Inference | Pretty Printing | Interpreter |
|:-------------------------|:--------------|:-------------|:---------------|:----------------|:------------|
| abs                      | yes           | yes          | yes            | yes             | no          |
| add                      | yes           | yes          | yes            | yes             | yes         |
| after_all                | yes           | yes          | no             | yes             | no          |
| all_gather               | no            | revisit      | no             | no              | no          |
| all_reduce               | no            | revisit      | revisit        | no              | no          |
| all_to_all               | no            | yes*         | yes*           | no              | no          |
| and                      | yes           | yes          | yes            | yes             | yes         |
| atan2                    | yes           | revisit      | yes            | yes             | no          |
| batch_norm_grad          | yes           | revisit      | yes            | no              | no          |
| batch_norm_inference     | yes           | revisit      | yes            | no              | no          |
| batch_norm_training      | yes           | revisit      | yes            | no              | no          |
| bitcast_convert          | yes           | yes          | infeasible     | yes             | no          |
| broadcast                | no            | yes*         | yes*           | yes             | no          |
| broadcast_in_dim         | yes           | yes          | infeasible     | yes             | no          |
| case                     | yes           | revisit      | yes            | no              | no          |
| cbrt                     | yes           | revisit      | yes            | yes             | no          |
| ceil                     | yes           | yes          | yes            | yes             | yes         |
| cholesky                 | yes           | yes          | yes            | yes             | no          |
| clamp                    | yes           | revisit      | yes            | yes             | no          |
| collective_permute       | no            | revisit      | revisit        | no              | no          |
| compare                  | yes           | yes          | yes            | yes             | no          |
| complex                  | yes           | yes          | yes            | yes             | no          |
| compute_reshape_shape    | no            | revisit      | no             | yes             | no          |
| concatenate              | yes           | yes          | yes            | yes             | no          |
| constant                 | yes           | yes          | yes            | yes             | yes         |
| convert                  | no            | yes*         | infeasible     | yes             | no          |
| convolution              | no            | yes*         | yes*           | revisit         | no          |
| cosine                   | yes           | yes          | yes            | yes             | yes         |
| count_leading_zeros      | yes           | yes          | yes            | yes             | no          |
| create_token             | no            | revisit      | no             | yes             | no          |
| cross-replica-sum        | no            | revisit      | revisit        | no              | no          |
| cstr_reshapable          | no            | revisit      | no             | yes             | no          |
| custom_call              | no            | revisit      | infeasible     | yes             | no          |
| divide                   | yes           | yes          | yes            | yes             | no          |
| dot                      | no            | revisit      | revisit        | yes             | no          |
| dot_general              | no            | yes*         | yes*           | no              | no          |
| dynamic_broadcast_in_dim | no            | revisit      | infeasible     | no              | no          |
| dynamic_conv             | no            | revisit      | no             | no              | no          |
| dynamic_gather           | no            | revisit      | revisit        | no              | no          |
| dynamic_iota             | no            | revisit      | infeasible     | yes             | no          |
| dynamic_pad              | no            | revisit      | no             | yes             | no          |
| dynamic_reshape          | no            | revisit      | infeasible     | yes             | no          |
| dynamic_slice            | yes           | revisit      | yes            | yes             | no          |
| dynamic_update_slice     | yes           | yes          | no             | yes             | no          |
| einsum                   | no            | revisit      | no             | no              | no          |
| exponential              | yes           | yes          | yes            | yes             | no          |
| exponential_minus_one    | yes           | yes          | yes            | yes             | no          |
| fft                      | yes           | revisit      | yes            | yes             | no          |
| floor                    | yes           | yes          | yes            | yes             | yes         |
| gather                   | yes           | yes          | yes            | no              | no          |
| get_dimension_size       | no            | revisit      | no             | yes             | no          |
| get_tuple_element        | yes           | yes          | yes            | yes             | no          |
| if                       | yes           | revisit      | yes            | no              | no          |
| imag                     | yes           | yes          | yes            | yes             | no          |
| infeed                   | no            | revisit      | no             | no              | no          |
| iota                     | yes           | yes          | infeasible     | yes             | yes         |
| is_finite                | yes           | yes          | yes            | yes             | no          |
| log                      | yes           | yes          | yes            | yes             | no          |
| log_plus_one             | yes           | yes          | yes            | yes             | no          |
| logistic                 | yes           | yes          | yes            | yes             | no          |
| map                      | yes           | revisit      | yes            | no              | no          |
| maximum                  | yes           | yes          | yes            | yes             | yes         |
| minimum                  | yes           | yes          | yes            | yes             | yes         |
| multiply                 | yes           | yes          | yes            | yes             | yes         |
| negate                   | yes           | yes          | yes            | yes             | yes         |
| not                      | yes           | yes          | yes            | yes             | yes         |
| optimization_barrier     | no            | revisit      | no             | yes             | no          |
| or                       | yes           | yes          | yes            | yes             | yes         |
| outfeed                  | no            | revisit      | no             | no              | no          |
| pad                      | yes           | yes          | yes            | yes             | no          |
| popcnt                   | yes           | yes          | yes            | yes             | no          |
| power                    | yes           | revisit      | yes            | yes             | no          |
| real                     | yes           | yes          | yes            | yes             | no          |
| real_dynamic_slice       | no            | revisit      | no             | yes             | no          |
| recv                     | no            | revisit      | no             | no              | no          |
| reduce                   | yes           | revisit      | yes            | revisit         | no          |
| reduce_precision         | no            | yes*         | yes*           | yes             | no          |
| reduce_scatter           | no            | revisit      | no             | no              | no          |
| reduce_window            | no            | yes*         | yes*           | no              | no          |
| remainder                | yes           | yes          | yes            | yes             | no          |
| replica_id               | yes           | yes          | yes            | yes             | no          |
| reshape                  | yes           | yes          | infeasible     | yes             | yes         |
| return                   | no            | revisit      | no             | yes             | no          |
| reverse                  | yes           | revisit      | yes            | yes             | no          |
| rng                      | yes           | yes          | yes            | yes             | no          |
| rng_bit_generator        | no            | yes*         | infeasible     | yes             | no          |
| round_nearest_afz        | yes           | yes          | yes            | yes             | no          |
| round_nearest_even       | yes           | yes          | yes            | yes             | no          |
| rsqrt                    | yes           | yes          | yes            | yes             | no          |
| scatter                  | yes           | revisit      | no             | no              | no          |
| select                   | yes           | yes          | yes            | yes             | no          |
| select_and_scatter       | no            | revisit      | no             | no              | no          |
| send                     | no            | revisit      | no             | no              | no          |
| set_dimension_size       | no            | yes*         | yes*           | yes             | no          |
| shift_left               | yes           | revisit      | yes            | yes             | no          |
| shift_right_arithmetic   | yes           | revisit      | yes            | yes             | no          |
| shift_right_logical      | yes           | revisit      | yes            | yes             | no          |
| sign                     | yes           | yes          | yes            | yes             | no          |
| sine                     | yes           | yes          | yes            | yes             | yes         |
| slice                    | yes           | yes          | yes            | no              | no          |
| sort                     | yes           | yes          | yes            | no              | no          |
| sqrt                     | yes           | yes          | yes            | yes             | no          |
| subtract                 | yes           | yes          | yes            | yes             | yes         |
| tanh                     | yes           | yes          | yes            | yes             | yes         |
| torch_index_select       | no            | revisit      | no             | no              | no          |
| trace                    | no            | revisit      | no             | yes             | no          |
| transpose                | yes           | yes          | yes            | yes             | yes         |
| triangular_solve         | yes           | revisit      | yes            | no              | no          |
| tuple                    | yes           | yes          | yes            | yes             | no          |
| unary_einsum             | no            | revisit      | no             | no              | no          |
| uniform_dequantize       | no            | yes*         | yes*           | yes             | no          |
| uniform_quantize         | no            | yes*         | infeasible     | yes             | no          |
| while                    | yes           | revisit      | yes            | revisit         | no          |
| xor                      | yes           | yes          | yes            | yes             | yes         |
