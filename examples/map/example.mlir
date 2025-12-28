// map 
cuda_tile.module @map_module {
    entry @example_kernel(%A: tile<ptr<f32>>, %B: tile<ptr<f32>>) {
        %t0 = make_token : token
        %bx, %by, %bz = get_tile_block_id : tile<i32>

        // print "Block ID: x=%d\n", %bx : tile<i32>
        // print "Block ID: y=%d\n", %by : tile<i32>
        // print "Block ID: z=%d\n", %bz : tile<i32>

        // 准备常量 10.0 的 tile
        %constant_10 = cuda_tile.constant <f32: 10.0> : !cuda_tile.tile<f32>
        %reshaped_constant_10 = reshape %constant_10 : tile<f32> -> tile<1xf32>
        %boardcasted_constant_10 = broadcast %reshaped_constant_10 : tile<1xf32> -> tile<8xf32>

        // 生成每 lane 的偏移：这里用 iota 得到 [0..7]
        %lane = iota : tile<8xi32>

        // 例：每个 block 处理 8 元，偏移 = blockIdx.x * 8 + lane
        %bx_s = reshape %bx : tile<i32> -> tile<1xi32>
        %bx_vec = broadcast %bx_s : tile<1xi32> -> tile<8xi32>
        %blk_stride_s = cuda_tile.constant <i32: 8> : !cuda_tile.tile<i32>
        %blk_stride_s_reshaped = reshape %blk_stride_s : tile<i32> -> tile<1xi32>
        %blk_stride = broadcast %blk_stride_s_reshaped : tile<1xi32> -> tile<8xi32>
        %blk_off = muli %bx_vec, %blk_stride : tile<8xi32>
        %elem_off = addi %blk_off, %lane : tile<8xi32>

        // 将标量指针扩展成 8 宽指针 tile 以便逐元素处理
        %a_ptr_1 = reshape %A : tile<ptr<f32>> -> tile<1xptr<f32>>
        %a_ptr = broadcast %a_ptr_1 : tile<1xptr<f32>> -> tile<8xptr<f32>>

        // offset op: ptr + elem_off（单位是元素大小；会用元素 bitwidth 计算字节）
        %aptrs_off = offset %a_ptr, %elem_off
                : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>

        // 加载 a ptr 数据，得到数据 tile 和 token
        %a, %t = load_ptr_tko weak %aptrs_off : tile<8xptr<f32>> -> tile<8xf32>, token

        %b_ptr_1 = reshape %B : tile<ptr<f32>> -> tile<1xptr<f32>>
        %b_ptr = broadcast %b_ptr_1 : tile<1xptr<f32>> -> tile<8xptr<f32>>

        %bptrs_off = offset %b_ptr, %elem_off
                : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>

        // 逐元素浮点加
        %sum = addf %a, %boardcasted_constant_10 : tile<8xf32>

        // print "Data: %f\n", %sum : tile<8xf32>

        // 写回，串到加载得到的 token
        %t3 = store_ptr_tko weak %bptrs_off, %sum token=%t
            : tile<8xptr<f32>>, tile<8xf32> -> token

        return
    }
}
