def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    # Khởi tạo giá trị ban đầu trước khi lặp
    dot_v_l = 0
    for vi, li in zip(v, L):
        dot_v_l += vi * li  # Nhân và cộng dồn

    dot_L_L = 0
    for li in L:
        dot_L_L += li ** 2  # Bình phương và cộng dồn

    if dot_L_L == 0:
        return -1

    c = dot_v_l / dot_L_L

    return [c * li for li in L]