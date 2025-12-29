cuda_tile.module @map2d {
  entry @example_kernel(%A: tile<ptr<bf16>>, %O: tile<ptr<bf16>>, %size: tile<i32>) {
    %t0 = make_token : token
    %bx, %by, %bz = get_tile_block_id : tile<i32>

    %c10 = cuda_tile.constant <bf16: 10.0> : !cuda_tile.tile<bf16>
    %c10r = reshape %c10 : tile<bf16> -> tile<1x1xbf16>
    %c10v = broadcast %c10r : tile<1x1xbf16> -> tile<4x4xbf16>

    %a_view = make_tensor_view %A, shape=[16, 16], strides=[16, 1]
            : tensor_view<16x16xbf16, strides=[16, 1]>
    %o_view = make_tensor_view %O, shape=[16, 16], strides=[16, 1]
            : tensor_view<16x16xbf16, strides=[16, 1]>

    %a = make_partition_view %a_view : partition_view<tile=(4x4), tensor_view<16x16xbf16, strides=[16,1]>>
    %o = make_partition_view %o_view : partition_view<tile=(4x4), tensor_view<16x16xbf16, strides=[16,1]>>

    %tile0, %res_token0 = load_view_tko weak %a[%bx, %by] token=%t0
      : partition_view<tile=(4x4), tensor_view<16x16xbf16, strides=[16,1]>>, tile<i32> -> tile<4x4xbf16>, token

    %tile1 = addf %tile0, %c10v : tile<4x4xbf16>

    %res_token1 = store_view_tko weak %tile1, %o[%bx, %by] token=%res_token0
          : tile<4x4xbf16>, partition_view<tile=(4x4), tensor_view<16x16xbf16, strides=[16,1]>>, tile<i32> -> token
    return
  }
}