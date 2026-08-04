# In-place binary post-ops: sum -> binary_add(dst), and more

## Overview

The Sum post-op in oneDNN is not like the other post-ops: it performs a binary
operation but requires no right-hand-side buffer, instead taking its input from
DST. This behavior can be replicated in all of the Binary post-ops, extending
the buffer semantics of Sum to all other Binaries and eventually deprecating
Sum.

The need for this, beside purely architectural, is also practical: many modern
LLMs like LLaMa, especially in OpenVINO, contain 3-GEMM subgraphs that can
benefit from a variant of Sum that performs multiplication instead of addition,
potentially saving dozens of % of E2E VRAM compared to regular binary_mul which
inevitably requires an RHS buffer that should exist in VRAM beside DST.

The alternative would be to add a new one-off Sum-like PO, let's call it Prod.
That would essentially duplicate the already very different codepath for Sum
in all of the primitives, whereas the changes to Binary would be benign at
worst, simultaneously bringing a lot more value (in the form of way more ops
than just Sum or Prod) to the table; the actual Binary post-op code isn't going
to change at all, the most labor-intensive part would be to determine the 0.1%
primitives implementations and configurations that alter DST before the final
write, and exclude them from the support matrix.

## Proposal

The new oneDNN interface for Binary should look like this:

```
// C++ API

void append_binary(algorithm aalgorithm, const memory::desc &src1_desc,
        const memory::desc &src2_desc, bool inplace);

void get_params_binary(int index, algorithm &aalgorithm,
        memory::desc &src1_desc, memory::desc &src2_desc, bool &inplace);

// C API

dnnl_status_t DNNL_API dnnl_post_ops_append_binary_v3(dnnl_post_ops_t post_ops,
        dnnl_alg_kind_t alg_kind, const_dnnl_memory_desc_t src1_desc,
        const_dnnl_memory_desc_t src2_desc, int inplace);

dnnl_status_t DNNL_API dnnl_post_ops_get_params_binary_v3(
        const_dnnl_post_ops_t post_ops, int index, dnnl_alg_kind_t *alg_kind,
        const_dnnl_memory_desc_t *src1_desc,
        const_dnnl_memory_desc_t *src2_desc, int *inplace);
```

That additional `int` means a boolean flag (for lack of `bool` in ANSI C) that
signals that the post-op being appended should be seen as an in-place Binary.


From a semantical standpoint, this is how the in-place Binaries differ from the
regular ones:

```
ACC = ACC + RHS — binary_add
ACC = ACC × RHS — binary_mul
ACC = ACC / RHS — binary_div
ACC = ACC + DST — Sum
ACC = ACC + DST — binary_add & inplace
ACC = ACC × DST — binary_mul & inplace
ACC = ACC / DST — binary_div & inplace
…
```

— where DST is the destination buffer, ACC is the accumulator register block
within the primitive, and RHS is the separate right-hand-side buffer used in
regular Binary post-ops.

When `inplace = true`, `user_src1_desc` should equal the memory descriptor of
`DNNL_ARG_DST` both in shape and in type, and `user_src2_desc` should equal
`nullptr`, otherwise the post-op is to be considered malformed.

## Implementation

Internally, the `inplace` flag should directly translate to a flag inside the
`binary_t` structure:

```
 struct binary_t {
     dnnl::impl::alg_kind_t alg;
+
+    // Binary ops with this flag set use DST as RHS which may interfere
+    // with some writes; declare this to let the parent primitive know.
+    bool src1_from_dst;
+
     // This is an unmodifiable user copy of attributes which is used in
     // caching mechanism. Not to be used internally.
     dnnl::impl::memory_desc_t user_src1_desc, user_src2_desc;
 
     // This is a modifiable copy of memory desc. It changes format kind
     // and tag of md in case user passed format_kind::any. To be used
     // everywhere internally.
     dnnl::impl::memory_desc_t src1_desc, src2_desc;
 };
```

The exact name of the flag is up to debate.

With this flag set, on execution the primitive that contains this as `i`-th
post-op should expect a reference to the `DNNL_ARG_DST` buffer passed to it in
`DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1`.

That way, no aspect of the existing Binary post-op pipeline is to be altered
in any way, since DST is, after all, just another regular buffer.

## Caveats

Not all of the oneDNN primitives can support in-place Binary post-ops — namely
those that write intermediate data to DST, e.g. when accumulating with atomics.
To guard against improper usage, a new `skip_mask_t` flag is to be introduced:

```
 enum class skip_mask_t : unsigned {
     none = 0,
     scales = 1u << 1,
     scales_groups = (unsigned)scales | (1u << 2),
     scales_data_type = (unsigned)scales | (1u << 3),
     zero_points = 1u << 4,
     zero_points_groups = (unsigned)zero_points | (1u << 5),
     zero_points_data_type = (unsigned)zero_points | (1u << 6),
     post_ops = 1u << 7,
     sum_dt = 1u << 8,
     rnn_data_qparams = 1u << 9,
     rnn_weights_qparams = 1u << 10,
     rnn_tparams = 1u << 11,
     rnn_weights_projection_qparams = 1u << 12,
     gpu_attr = 1u << 13,
     accumulation_mode = 1u << 14,
     fpmath_mode = 1u << 15,
     dropout = 1u << 16,
     rounding_mode = 1u << 17,
     precomputed_reductions = 1u << 18,
+    inplace_binary_post_ops = (unsigned)post_ops | (1u << 19),
 };
```

That way in-place post-ops become an opt-in feature, so by default no primitive
would accept them unless the implementation allows it explicitly.

## Interface for benchdnn

The benchdnn interface for in-place Binary post-ops is fairly strightforward.
Just adding an `inplace` keyword after the Binary alg name should be enough:

```
… --attr-post-ops=mul:inplace …
```

## Conclusion

In-place Binary post-ops are a powerful tool available at virtually no cost to
the library; they enable significant memory savings and new approaches to DNN
graph optimization.

After introduction of in-place Binary post-ops, the Sum post-op can be
considered deprecated.

