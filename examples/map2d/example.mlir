cuda_tile.module @map2d {
  entry @example_kernel(%A: tile<ptr<f32>>, %O: tile<ptr<f32>>, %size: tile<i32>) {
    %t0 = make_token : token
    %bx, %by, %bz = get_tile_block_id : tile<i32>

    %c10 = cuda_tile.constant <f32: 10.0> : !cuda_tile.tile<f32>
    %c10r = reshape %c10 : tile<f32> -> tile<1x1xf32>
    %c10v = broadcast %c10r : tile<1x1xf32> -> tile<4x4xf32>

    %a_view = make_tensor_view %A, shape=[16, 16], strides=[16, 1]
            : tensor_view<16x16xf32, strides=[16, 1]>
    %o_view = make_tensor_view %O, shape=[16, 16], strides=[16, 1]
            : tensor_view<16x16xf32, strides=[16, 1]>

    %a = make_partition_view %a_view : partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>
    %o = make_partition_view %o_view : partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>

    %tile0, %res_token0 = load_view_tko weak %a[%bx, %by] token=%t0
      : partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> tile<4x4xf32>, token

    %tile1 = addf %tile0, %c10v : tile<4x4xf32>

    %res_token1 = store_view_tko weak %tile1, %o[%bx, %by] token=%res_token0
          : tile<4x4xf32>, partition_view<tile=(4x4), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> token
    return
  }
}

// cuda_tile.module @map2d {
//   entry @example_kernel(%A: tile<ptr<f32>>, %O: tile<ptr<f32>>, %size: tile<i32>) {
//     %t0 = make_token : token
//     %bx, %by, %bz = get_tile_block_id : tile<i32>
//     %lane = iota : tile<16xi32>

//     // 常量与广播（固定 4x4 tile，行宽 16）
//     %c4  = cuda_tile.constant <i32: 4>  : !cuda_tile.tile<i32>
//     %c10 = cuda_tile.constant <f32: 10.0> : !cuda_tile.tile<f32>
//     %c4r = reshape %c4  : tile<i32> -> tile<1xi32>
//     %c4v  = broadcast %c4r : tile<1xi32> -> tile<16xi32>
//     %sizer = reshape %size : tile<i32> -> tile<1xi32>
//     %sizev = broadcast %sizer : tile<1xi32> -> tile<16xi32>
//     %c10r = reshape %c10 : tile<f32> -> tile<1xf32>
//     %c10v = broadcast %c10r : tile<1xf32> -> tile<16xf32>
//     %c16 = cuda_tile.constant <i32: 16> : !cuda_tile.tile<i32>
//     %c16r = reshape %c16 : tile<i32> -> tile<1xi32>
//     %c16v = broadcast %c16r : tile<1xi32> -> tile<16xi32>

//     // block 起点行/列（4×4 tile）
//     %bxr = reshape %bx : tile<i32> -> tile<1xi32>
//     %bxv = broadcast %bxr : tile<1xi32> -> tile<16xi32>
//     %byr = reshape %by : tile<i32> -> tile<1xi32>
//     %byv = broadcast %byr : tile<1xi32> -> tile<16xi32>
//     %tile_row0 = muli %byv, %c4v : tile<16xi32>
//     %tile_col0 = muli %bxv, %c4v : tile<16xi32>

//     // lane 内部行/列
//     %lane_row = divi %lane, %c4v signed : tile<16xi32>
//     %lane_col = remi %lane, %c4v signed : tile<16xi32>

//     // 全局行/列
//     %row = addi %tile_row0, %lane_row : tile<16xi32>
//     %col = addi %tile_col0, %lane_col : tile<16xi32>

//     // 线性偏移 = row*16 + col（固定行宽以减少计算）
//     %idx_row = muli %row, %c16v : tile<16xi32>
//     %elem_off = addi %idx_row, %col : tile<16xi32>

//     // 指针偏移并加载（输入）
//     %A1 = reshape %A : tile<ptr<f32>> -> tile<1xptr<f32>>
//     %A16 = broadcast %A1 : tile<1xptr<f32>> -> tile<16xptr<f32>>
//     %Ap = offset %A16, %elem_off : tile<16xptr<f32>>, tile<16xi32> -> tile<16xptr<f32>>
//     %a, %t1 = load_ptr_tko weak %Ap : tile<16xptr<f32>> -> tile<16xf32>, token

//     // 计算与 select（越界则保持原值）
//     %add = addf %a, %c10v : tile<16xf32>

//     // 输出指针与偏移
//     %O1 = reshape %O : tile<ptr<f32>> -> tile<1xptr<f32>>
//     %O16 = broadcast %O1 : tile<1xptr<f32>> -> tile<16xptr<f32>>
//     %Op = offset %O16, %elem_off : tile<16xptr<f32>>, tile<16xi32> -> tile<16xptr<f32>>

//     // 写回到输出
//     %t2 = store_ptr_tko weak %Op, %add token=%t1
//       : tile<16xptr<f32>>, tile<16xf32> -> token
//     return
//   }
// }
