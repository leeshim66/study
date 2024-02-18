from collections import deque

def solution(x, y, n):
    dist = [0] * 1000001
    q = deque()
    dist[x] = 1
    q.append(x)

    while q:
        x = q.popleft()
        if 0 <= x + n <= 1000000 and dist[x + n] == 0:
            dist[x + n] = dist[x] + 1
            q.append(x + n)
        if 0 <= x * 2 <= 1000000 and dist[x * 2] == 0:
            dist[x * 2] = dist[x] + 1
            q.append(x * 2)
        if 0 <= x * 3 <= 1000000 and dist[x * 3] == 0:
            dist[x * 3] = dist[x] + 1
            q.append(x * 3)

    return dist[y]-1

x = 10
y = 40
n = 5
answer = 2