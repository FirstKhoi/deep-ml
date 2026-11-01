def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	    
    if len(a[0]) != len(b):
        return -1

    result = []

    for row in a:
        new_row = []

        for j in range(len(b[0])):
            value = sum(
                row[k] * b[k][j]
                for k in range(len(b))
            )

            new_row.append(value)

        result.append(new_row)

    return result