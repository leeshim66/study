def solution(arr):
    answer = []
    div = len(arr)
    vst = []
    a, b = 0, 0
    zerocnt, onecnt = 0, 0
    for i in range(len(arr)):
        tmp = []
        for j in range(len(arr)):
            tmp = [0] * len(arr)
        vst.append(tmp)

    while(div > 0):
        sm = 0
        if(vst[a][b] == 0):
            for i in range(div):
                sm += sum(arr[a + i][b:b + div])

            if(sm == div ** 2):
                for i in range(div):
                    for j in range(div):
                        vst[a + i][b + j] = 1
                onecnt += 1
            if(sm == 0):
                zerocnt += 1
                for i in range(div):
                    for j in range(div):
                        vst[a + i][b + j] = 1

        if(b + div >= len(arr)):
            a += div
            b = 0
        else:
            b += div

        if(a >= len(arr)):
            a, b = 0, 0
            div = div // 2

    answer = [zerocnt,onecnt]
    return answer

arr = [[1,1,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1]]
answer = [4,9]