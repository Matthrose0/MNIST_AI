# später für Einzelteile wenn keine komplexen berechnungen mehr mit numpy,...
class Utils:
    @staticmethod
    def transpose(inputList: list[list[int]]):
        """
        Matrix transposing
        """
        return [[row[i] for row in inputList] for i in range(Utils.searchForLowestRowCount(inputList))]
    
    @staticmethod
    def searchForLowestRowCount(inputList: list[list[int]]):
        """
        searching an Matrix for the smallest amount of elements in a list
        """
        return min(len(row) for row in inputList) if inputList else 0


test = [[1, 2, 3, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10,11,12]]
print(Utils.transpose(test))