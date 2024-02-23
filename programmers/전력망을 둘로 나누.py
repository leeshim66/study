def solution(n, wires):
    answer = -1
    saveWires = wires.copy()
    countList = []
    count = 0

    for comb in wires.copy():
        saveWires.remove(comb)
        setl = set([comb[0]])
        previous = set()
        while (setl != previous) :
            previous = setl.copy()
            for wire in saveWires.copy():
                if (setl & set(wire)):
                    setl = setl | set(wire)
        countList.append(abs((n - 2*len(setl))))
        count = 0
        saveWires = wires.copy()

    answer = min(countList)
    return answer

n = 9
wires = [[1,3],[2,3],[3,4],[4,5],[4,6],[4,7],[7,8],[7,9]]
answer = 3