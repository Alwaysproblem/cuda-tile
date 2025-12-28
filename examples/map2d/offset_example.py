a = [i for i in range(256)]

b = [[f"{i}, {j}: {a[i*16+j]}" for j in range(16)] for i in range(16) ]
print(b)

for by in range(4):
    for bx in range(4):
        lane = [i for i in range(16)]
        lane_row = [i // 4 + by * 4 for i in lane]
        lane_col = [i % 4 + bx * 4 for i in lane]
        elem_off = [lane_row[i] * 16 + lane_col[i] for i in range(16)]
        print(f"Block ({bx}, {by}): {bx * 4}, {by * 4}, offset: {elem_off}")