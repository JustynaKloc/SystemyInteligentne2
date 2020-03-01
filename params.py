from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class FuzzyInputVariable_3Trapezoids:
    w = 0.0  # type: float
    c = 0.0  # type: float
    fl = 0.0  # type: float
    fr = 0.0  # type: float
    name = None  # type: str
    labels = None  # type: List[str]
    n_functions = 3  # type: int
    n_params = 4  # type: int

    def __init__(self, center: float, kernelWidth: float, fuzzyLeftWidth: float, fuzzyRightWidth, name: str,
                 labels: List[str]):
        self.set(center, kernelWidth, fuzzyLeftWidth, fuzzyRightWidth)
        self.name = name
        self.labels = list(labels)

    def set(self, center: float, kernelWidth: float, fuzzyLeftWidth: float, fuzzyRightWidth: float):
        self.c = center
        self.w = kernelWidth
        self.fl = fuzzyLeftWidth
        self.fr = fuzzyRightWidth

    def get(self) -> List[float]:
        return [self.c, self.w, self.fl, self.fr]

    def getFunctionsList(self, leftBound=-10, rightBound=10):
        assert leftBound < self.c - self.fl <= self.c + self.fr < rightBound
        w1 = self.c - self.w / 2 - self.fl - leftBound
        mf1 = [self.c - self.w / 2 - self.fl - w1 / 2, w1, 0, self.fl]

        w3 = rightBound - (self.c + self.w / 2 + self.fr)
        mf3 = [self.c + self.w / 2 + self.fr + w3 / 2, w3, self.fr, 0]
        return [mf1, list(self.get()), mf3]

    def show3DX(self, ax):
        self.__show3d(ax, 0)

    def show3DY(self, ax):
        self.__show3d(ax, 1)

    def __show3d(self, ax=None, axis=None):

        ax = ax or plt
        assert axis in [0, 1]  # 0-os x, 1-os y

        arg = np.arange(-1.5, 1.5, 0.01)
        offset = np.ones((len(arg))) * 1.5
        fuzz = self.fuzzify(arg)

        for i in range(0, 3):
            if axis == 0:
                ax.plot(arg, offset, fuzz[:, i], label=f"{self.name}: {self.labels[i]}")
            else:
                ax.plot(-offset, arg, fuzz[:, i], label=f"{self.name}: {self.labels[i]}")

    def show(self, x=None):
        xmargin = 0.1
        ymargin = 0.1

        if x is None:
            x = np.arange(0, 3, 0.01)
            # plt.xlim([-1.5-xmargin, 1.5+xmargin])

        y = self.fuzzify(x)

        colors = ['#ff0000', '#ff0000', '#ff0000']
        ls = ['solid', 'dashed', 'dotted']
        for i in range(0, 3):
            plt.plot(x, y[:, i], label=f"{self.name}: {self.labels[i]}", color=colors[i], linestyle=ls[i])

        plt.title(f"Zmienna lingwistyczna {self.name}")

        plt.ylim([0 - ymargin, 1 + ymargin])

    def showRegions(self, ax, orientation):

        begin = -5

        if orientation == 0:
            ax.add_patch(
                Rectangle((begin, self.c - self.w / 2 - self.fl), 100, self.fl, fill=False, hatch='/', color='r'))
            ax.add_patch(Rectangle((begin, self.c + self.w / 2), 100, self.fr, fill=False, hatch='/', color='r'))
            return self.c + self.w / 2 + self.fr / 2, self.c - self.w / 2 - self.fl / 2
        else:
            ax.add_patch(
                Rectangle((self.c - self.w / 2 - self.fl, begin), self.fl, 100, fill=False, hatch='/', color='b'))
            ax.add_patch(Rectangle((self.c + self.w / 2, begin), self.fr, 100, fill=False, hatch='/', color='b'))
            return self.c + self.w / 2 + self.fr / 2, self.c - self.w / 2 - self.fl / 2

    def fuzzify(self, x: Union[float, np.ndarray]) -> np.ndarray:

        # wylicz pozycje
        x1 = self.c - 0.5 * self.w - self.fl
        x2 = self.c - 0.5 * self.w
        x3 = self.c + 0.5 * self.w
        x4 = self.c + 0.5 * self.w + self.fr

        dx21 = x2 - x1
        dx34 = x3 - x4

        if dx21 == 0:
            dx21 = 0.01
        else:
            dx21 = np.sign(dx21) * 0.01 if np.abs(dx21) < 0.01 else dx21

        if dx34 == 0:
            dx34 = -0.01
        else:
            dx34 = np.sign(dx34) * 0.01 if np.abs(dx34) < 0.01 else dx34

        # wyznacz zbocza funkcji
        ya = (x - x1) / dx21
        yb = (x - x4) / dx34

        # wyznacz wartości przynależności
        yl = np.clip(1 - ya, 0, 1)
        yr = np.clip(1 - yb, 0, 1)
        yc = np.clip(np.minimum(ya, yb), 0, 1)

        # zwróć wszystko w formie macierzy: kolumna = przynależności do danej wartości rozmytej
        output = np.column_stack((yl, yc, yr))
        return output


class FuzzyInputVariable_2Sigmoids:
    # # fw = 0.0
    # # c = 0.0
    # # name = None
    # # labels = None
    n_functions = 2

    # # n_params = 2

    def __init__(self, center: float, fuzzyWidth: float, name: str, labels: List[str]):
        self.c = center
        self.fw = fuzzyWidth
        self.name = name
        self.labels = list(labels)

    def set(self, center: float, fuzzyWidth: float):
        self.c = center
        self.fw = fuzzyWidth

    def get(self) -> List[float]:
        return [self.c, self.fw]

    # def getFunctionsList(self, leftBound=-10, rightBound=10):
    #     assert leftBound < self.c - self.fw <= self.c + self.fw < rightBound
    #     w1 = self.c - self.fw / 2 - leftBound
    #     mf1 = [self.c - self.fw / 2 - w1 / 2, w1, 0, self.fw]
    #
    #     w2 = rightBound - (self.c + self.fw / 2)
    #     mf2 = [self.c + self.fw / 2 + w2 / 2, w2, self.fw, 0]
    #     return [mf1, mf2]

    def show(self, x=None):
        xmargin = 0.1
        ymargin = 0.1

        if x is None:
            x = np.arange(-10, 20, 0.01)
            # plt.xlim([0-xmargin, 1+xmargin])

        y = self.fuzzify(x)

        plt.plot(x, y[:, 0], label=self.labels[0])
        plt.plot(x, y[:, 1], label=self.labels[1])

        plt.title(f"Zmienna lingwistyczna {self.name}")
        plt.ylim([0 - ymargin, 1 + ymargin])

    # def showRegions(self, ax, orientation):
    #     begin = -5
    #
    #     if orientation == 0:
    #         ax.add_patch(Rectangle((begin, self.c - self.fw / 2), 100, self.fw, fill=False, hatch='/', color='r'))
    #         return [self.c]
    #     else:
    #         ax.add_patch(Rectangle((self.c - self.fw / 2, begin), self.fw, 100, fill=False, hatch='/', color='b'))
    #         return [self.c]

    def fuzzify(self, x: Union[float, np.ndarray]) -> np.ndarray:
        # wylicz pozycje
        x1 = self.fw
        x2 = self.c

        # # wyznacz wartości przynależności
        yr = 1 / (1 + np.exp(-x1 * (x - x2)))
        yl = 1 / (1 + np.exp(-x1 * (-(x - 2 * x2) - x2)))

        # # wyznacz zbocza funkcji
        # y = (x - x2) / dx12
        #
        # # wyznacz wartości przynależności
        # yl = np.clip(y, 0, 1)
        # yr = np.clip(1 - y, 0, 1)

        # zwróć wszystko w formie macierzy: kolumna = przynależności do danej wartości rozmytej
        output = np.column_stack((yl, yr))
        return output


class FuzzyInputVariable_2Trapezoids:
    fw = 0.0  # type: float
    c = 0.0  # type: float
    name = None  # type: str
    labels = None  # type: List[str]
    n_functions = 2  # type: int
    n_params = 2  # type: int

    def __init__(self, center: float, fuzzyWidth: float, name: str, labels: List[str]):
        self.c = center
        self.fw = fuzzyWidth
        self.name = name
        self.labels = list(labels)

    def set(self, center: float, fuzzyWidth: float):
        self.c = center
        self.fw = fuzzyWidth

    def get(self) -> List[float]:
        return [self.c, self.fw]

    def getFunctionsList(self, leftBound=-10, rightBound=10):
        assert leftBound < self.c - self.fw <= self.c + self.fw < rightBound
        w1 = self.c - self.fw / 2 - leftBound
        mf1 = [self.c - self.fw / 2 - w1 / 2, w1, 0, self.fw]

        w2 = rightBound - (self.c + self.fw / 2)
        mf2 = [self.c + self.fw / 2 + w2 / 2, w2, self.fw, 0]
        return [mf1, mf2]

    def show(self, x=None):
        xmargin = 0.1
        ymargin = 0.1

        if x is None:
            x = np.arange(0, 3, 0.01)
            # plt.xlim([0-xmargin, 1+xmargin])

        y = self.fuzzify(x)

        plt.plot(x, y[:, 0], label=self.labels[0])
        plt.plot(x, y[:, 1], label=self.labels[1])

        plt.title(f"Zmienna lingwistyczna {self.name}")

        plt.ylim([0 - ymargin, 1 + ymargin])

    def showRegions(self, ax, orientation):

        begin = -5

        if orientation == 0:
            ax.add_patch(Rectangle((begin, self.c - self.fw / 2), 100, self.fw, fill=False, hatch='/', color='r'))
            return [self.c]
        else:
            ax.add_patch(Rectangle((self.c - self.fw / 2, begin), self.fw, 100, fill=False, hatch='/', color='b'))
            return [self.c]

    def fuzzify(self, x: Union[float, np.ndarray]) -> np.ndarray:

        # wylicz pozycje
        x1 = self.c - 0.5 * self.fw
        x2 = self.c + 0.5 * self.fw

        dx12 = x1 - x2

        if dx12 == 0:
            dx12 = -0.01
        else:
            dx12 = np.sign(dx12) * 0.01 if np.abs(dx12) < 0.01 else dx12

        # wyznacz zbocza funkcji
        y = (x - x2) / dx12

        # wyznacz wartości przynależności
        yl = np.clip(y, 0, 1)
        yr = np.clip(1 - y, 0, 1)

        # zwróć wszystko w formie macierzy: kolumna = przynależności do danej wartości rozmytej
        output = np.column_stack((yl, yr))
        return output


class FuzzyInputVariable_List_Trapezoids:
    w = 0.0  # type: float
    c = 0.0  # type: float
    fl = 0.0  # type: float
    fr = 0.0  # type: float
    name = None  # type: str
    labels = None  # type: List[str]

    def __init__(self, listCWFLFR: List[List[float]], name: str, labels: List[str]):
        self.name = name
        self.labels = list(labels)

        self.n_functions = len(listCWFLFR)  # type: int
        self.n_params = 4 * self.n_functions  # type: int

        self.functionsList = listCWFLFR  # type: List[double]

    def set(self, *function_list):
        self.functionsList = function_list

    def get(self) -> List[float]:
        return self.functionsList

    def getFunctionsList(self):
        return self.functionsList

    def show3DX(self, ax):
        self.__show3d(ax, 0)

    def show3DY(self, ax):
        self.__show3d(ax, 1)

    # TODO: przestestować    
    def __show3d(self, ax=None, axis=None):

        ax = ax or plt
        assert axis in [0, 1]  # 0-os x, 1-os y

        arg = np.arange(-1.5, 1.5, 0.01)
        offset = np.ones((len(arg))) * 1.5
        fuzz = self.fuzzify(arg)

        for i in range(0, self.n_functions):
            if axis == 0:
                ax.plot(arg, offset, fuzz[:, i], label=f"{self.name}: {self.labels[i]}")
            else:
                ax.plot(-offset, arg, fuzz[:, i], label=f"{self.name}: {self.labels[i]}")

    def show(self, x=None):
        xmargin = 0.1
        ymargin = 0.1

        if x is None:
            x = np.arange(-1.5, 1.5, 0.01)
            plt.xlim([-1.5 - xmargin, 1.5 + xmargin])

        y = self.fuzzify(x)

        # colors = ['#ff0000','#ff0000','#ff0000','#ff0000']
        # ls = ['solid','dashed','dotted','dashdot']
        for i in range(0, self.n_functions):
            # plt.plot(x, y[:, i], label=f"{self.name}: {self.labels[i]}", color=colors[i], linestyle=ls[i])
            plt.plot(x, y[:, i], label=f"{self.name}: {self.labels[0]}")

        plt.title(f"Zmienna lingwistyczna {self.name}")

        plt.ylim([0 - ymargin, 1 + ymargin])

    def fuzzifyOneLinguisticValue(self, x: Union[float, np.ndarray], idx: int) -> np.ndarray:

        c, w, fl, fr = self.functionsList[idx];
        w = abs(w)
        fl = abs(fl)
        fr = abs(fr)

        # wylicz pozycje
        x1 = c - 0.5 * w - fl
        x2 = c - 0.5 * w
        x3 = c + 0.5 * w
        x4 = c + 0.5 * w + fr

        dx21 = x2 - x1
        dx34 = x3 - x4

        if dx21 == 0:
            dx21 = 0.01
        else:
            dx21 = np.sign(dx21) * 0.01 if np.abs(dx21) < 0.01 else dx21

        if dx34 == 0:
            dx34 = -0.01
        else:
            dx34 = np.sign(dx34) * 0.01 if np.abs(dx34) < 0.01 else dx34

        # wyznacz zbocza funkcji
        ya = (x - x1) / dx21
        yb = (x - x4) / dx34

        # wyznacz wartości przynależności
        yc = np.clip(np.minimum(ya, yb), 0, 1)

        return yc

    def fuzzify(self, x: Union[float, np.ndarray]) -> np.ndarray:

        output = [self.fuzzifyOneLinguisticValue(x, i) for i in range(self.n_functions)]

        return np.array(output).T
