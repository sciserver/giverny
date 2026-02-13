########################################################################
#
#  Copyright 2024 Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: turbulence@pha.jhu.edu
# Website: http://turbulence.pha.jhu.edu/
#
########################################################################

"""
calculate the interpolation weights for the non-constant step grids.
    - channel, channel5200, transition_bl datasets.
"""
import math
import numpy as np

class channel_y_interpolation_weights():
    def __init__(self, lagrange_order):
        self.lagrange_order = lagrange_order
        self.y_values = self.load_channel_y_grid()
        self.barycentric_weights = self.calculate_barycentric_weights()

    def load_channel_y_grid(self):
        """
        load the channel y-grid point values into a numpy array.
        """
        ys = """
        -9.999999999999999e-01
        -9.999834694070191e-01
        -9.999449902371761e-01
        -9.998791453391053e-01
        -9.997805185994304e-01
        -9.996436951481997e-01
        -9.994632615642732e-01
        -9.992338060806554e-01
        -9.989664493827454e-01
        -9.986612016114569e-01
        -9.983180743449220e-01
        -9.979370805980530e-01
        -9.975182348220475e-01
        -9.970615529038419e-01
        -9.965670521655072e-01
        -9.960347513635933e-01
        -9.954646706884165e-01
        -9.948568317632942e-01
        -9.942112576437249e-01
        -9.935279728165133e-01
        -9.928070031988414e-01
        -9.920483761372862e-01
        -9.912521204067818e-01
        -9.904182662095278e-01
        -9.895468451738442e-01
        -9.886378903529718e-01
        -9.876914362238178e-01
        -9.867075186856491e-01
        -9.856861750587294e-01
        -9.846274440829048e-01
        -9.835313659161335e-01
        -9.823979821329635e-01
        -9.812273357229543e-01
        -9.800194710890480e-01
        -9.787744340458836e-01
        -9.774922718180602e-01
        -9.761730330383451e-01
        -9.748167677458299e-01
        -9.734235273840317e-01
        -9.719933647989422e-01
        -9.705263342370233e-01
        -9.690224913431497e-01
        -9.674818931584975e-01
        -9.659045981183816e-01
        -9.642906660500385e-01
        -9.626401581703575e-01
        -9.609531370835583e-01
        -9.592296667788167e-01
        -9.574698126278376e-01
        -9.556736413823750e-01
        -9.538412211717003e-01
        -9.519726215000186e-01
        -9.500679132438312e-01
        -9.481271686492487e-01
        -9.461504613292497e-01
        -9.441378662608887e-01
        -9.420894597824527e-01
        -9.400053195905654e-01
        -9.378855247372397e-01
        -9.357301556268803e-01
        -9.335392940132328e-01
        -9.313130229962833e-01
        -9.290514270191066e-01
        -9.267545918646627e-01
        -9.244226046525438e-01
        -9.220555538356686e-01
        -9.196535291969289e-01
        -9.172166218457825e-01
        -9.147449242147984e-01
        -9.122385300561506e-01
        -9.096975344380619e-01
        -9.071220337411980e-01
        -9.045121256550118e-01
        -9.018679091740380e-01
        -8.991894845941382e-01
        -8.964769535086964e-01
        -8.937304188047661e-01
        -8.909499846591674e-01
        -8.881357565345354e-01
        -8.852878411753202e-01
        -8.824063466037383e-01
        -8.794913821156746e-01
        -8.765430582765373e-01
        -8.735614869170644e-01
        -8.705467811290808e-01
        -8.674990552612101e-01
        -8.644184249145361e-01
        -8.613050069382191e-01
        -8.581589194250628e-01
        -8.549802817070359e-01
        -8.517692143507448e-01
        -8.485258391528615e-01
        -8.452502791355032e-01
        -8.419426585415662e-01
        -8.386031028300133e-01
        -8.352317386711151e-01
        -8.318286939416456e-01
        -8.283940977200313e-01
        -8.249280802814557e-01
        -8.214307730929177e-01
        -8.179023088082450e-01
        -8.143428212630627e-01
        -8.107524454697166e-01
        -8.071313176121525e-01
        -8.034795750407501e-01
        -7.997973562671140e-01
        -7.960848009588195e-01
        -7.923420499341149e-01
        -7.885692451565806e-01
        -7.847665297297438e-01
        -7.809340478916512e-01
        -7.770719450093977e-01
        -7.731803675736124e-01
        -7.692594631929024e-01
        -7.653093805882540e-01
        -7.613302695873916e-01
        -7.573222811190946e-01
        -7.532855672074728e-01
        -7.492202809662000e-01
        -7.451265765927062e-01
        -7.410046093623291e-01
        -7.368545356224248e-01
        -7.326765127864361e-01
        -7.284706993279236e-01
        -7.242372547745537e-01
        -7.199763397020476e-01
        -7.156881157280909e-01
        -7.113727455062038e-01
        -7.070303927195706e-01
        -7.026612220748322e-01
        -6.982653992958381e-01
        -6.938430911173605e-01
        -6.893944652787704e-01
        -6.849196905176745e-01
        -6.804189365635158e-01
        -6.758923741311349e-01
        -6.713401749142954e-01
        -6.667625115791711e-01
        -6.621595577577967e-01
        -6.575314880414823e-01
        -6.528784779741906e-01
        -6.482007040458788e-01
        -6.434983436858043e-01
        -6.387715752557943e-01
        -6.340205780434813e-01
        -6.292455322555019e-01
        -6.244466190106620e-01
        -6.196240203330663e-01
        -6.147779191452152e-01
        -6.099084992610652e-01
        -6.050159453790577e-01
        -6.001004430751129e-01
        -5.951621787955912e-01
        -5.902013398502206e-01
        -5.852181144049931e-01
        -5.802126914750259e-01
        -5.751852609173937e-01
        -5.701360134239264e-01
        -5.650651405139760e-01
        -5.599728345271530e-01
        -5.548592886160300e-01
        -5.497246967388160e-01
        -5.445692536519992e-01
        -5.393931549029601e-01
        -5.341965968225542e-01
        -5.289797765176648e-01
        -5.237428918637272e-01
        -5.184861414972231e-01
        -5.132097248081458e-01
        -5.079138419324380e-01
        -5.025986937444001e-01
        -4.972644818490711e-01
        -4.919114085745817e-01
        -4.865396769644800e-01
        -4.811494907700297e-01
        -4.757410544424823e-01
        -4.703145731253217e-01
        -4.648702526464831e-01
        -4.594082995105464e-01
        -4.539289208909028e-01
        -4.484323246218968e-01
        -4.429187191909435e-01
        -4.373883137306198e-01
        -4.318413180107326e-01
        -4.262779424303618e-01
        -4.206983980098800e-01
        -4.151028963829484e-01
        -4.094916497884894e-01
        -4.038648710626363e-01
        -3.982227736306606e-01
        -3.925655714988764e-01
        -3.868934792465233e-01
        -3.812067120176273e-01
        -3.755054855128403e-01
        -3.697900159812586e-01
        -3.640605202122202e-01
        -3.583172155270825e-01
        -3.525603197709786e-01
        -3.467900513045550e-01
        -3.410066289956883e-01
        -3.352102722111846e-01
        -3.294012008084577e-01
        -3.235796351271907e-01
        -3.177457959809780e-01
        -3.118999046489499e-01
        -3.060421828673796e-01
        -3.001728528212723e-01
        -2.942921371359380e-01
        -2.884002588685473e-01
        -2.824974414996703e-01
        -2.765839089248003e-01
        -2.706598854458616e-01
        -2.647255957627011e-01
        -2.587812649645658e-01
        -2.528271185215648e-01
        -2.468633822761177e-01
        -2.408902824343875e-01
        -2.349080455577012e-01
        -2.289168985539562e-01
        -2.229170686690132e-01
        -2.169087834780773e-01
        -2.108922708770653e-01
        -2.048677590739623e-01
        -1.988354765801652e-01
        -1.927956522018153e-01
        -1.867485150311196e-01
        -1.806942944376616e-01
        -1.746332200597008e-01
        -1.685655217954626e-01
        -1.624914297944183e-01
        -1.564111744485555e-01
        -1.503249863836389e-01
        -1.442330964504632e-01
        -1.381357357160963e-01
        -1.320331354551152e-01
        -1.259255271408336e-01
        -1.198131424365220e-01
        -1.136962131866207e-01
        -1.075749714079459e-01
        -1.014496492808893e-01
        -9.532047914061102e-02
        -8.918769346822775e-02
        -8.305152488199409e-02
        -7.691220612847960e-02
        -7.076997007374050e-02
        -6.462504969448707e-02
        -5.847767806924677e-02
        -5.232808836952347e-02
        -4.617651385095325e-02
        -4.002318784445690e-02
        -3.386834374738965e-02
        -2.771221501468831e-02
        -2.155503515001626e-02
        -1.539703769690663e-02
        -9.238456229903897e-03
        -3.079524345704338e-03
        3.079524345704338e-03
        9.238456229903897e-03
        1.539703769690663e-02
        2.155503515001626e-02
        2.771221501468831e-02
        3.386834374738965e-02
        4.002318784445690e-02
        4.617651385095325e-02
        5.232808836952347e-02
        5.847767806924677e-02
        6.462504969448710e-02
        7.076997007374054e-02
        7.691220612847967e-02
        8.305152488199419e-02
        8.918769346822787e-02
        9.532047914061116e-02
        1.014496492808894e-01
        1.075749714079461e-01
        1.136962131866209e-01
        1.198131424365222e-01
        1.259255271408338e-01
        1.320331354551154e-01
        1.381357357160965e-01
        1.442330964504634e-01
        1.503249863836391e-01
        1.564111744485556e-01
        1.624914297944185e-01
        1.685655217954628e-01
        1.746332200597010e-01
        1.806942944376618e-01
        1.867485150311198e-01
        1.927956522018154e-01
        1.988354765801654e-01
        2.048677590739625e-01
        2.108922708770654e-01
        2.169087834780773e-01
        2.229170686690132e-01
        2.289168985539561e-01
        2.349080455577011e-01
        2.408902824343873e-01
        2.468633822761175e-01
        2.528271185215647e-01
        2.587812649645656e-01
        2.647255957627010e-01
        2.706598854458615e-01
        2.765839089248003e-01
        2.824974414996702e-01
        2.884002588685473e-01
        2.942921371359380e-01
        3.001728528212723e-01
        3.060421828673796e-01
        3.118999046489499e-01
        3.177457959809780e-01
        3.235796351271907e-01
        3.294012008084577e-01
        3.352102722111846e-01
        3.410066289956883e-01
        3.467900513045550e-01
        3.525603197709786e-01
        3.583172155270825e-01
        3.640605202122202e-01
        3.697900159812586e-01
        3.755054855128403e-01
        3.812067120176273e-01
        3.868934792465233e-01
        3.925655714988764e-01
        3.982227736306606e-01
        4.038648710626363e-01
        4.094916497884894e-01
        4.151028963829484e-01
        4.206983980098800e-01
        4.262779424303618e-01
        4.318413180107326e-01
        4.373883137306198e-01
        4.429187191909435e-01
        4.484323246218969e-01
        4.539289208909028e-01
        4.594082995105465e-01
        4.648702526464832e-01
        4.703145731253218e-01
        4.757410544424824e-01
        4.811494907700299e-01
        4.865396769644801e-01
        4.919114085745818e-01
        4.972644818490712e-01
        5.025986937444002e-01
        5.079138419324382e-01
        5.132097248081460e-01
        5.184861414972232e-01
        5.237428918637274e-01
        5.289797765176649e-01
        5.341965968225543e-01
        5.393931549029602e-01
        5.445692536519993e-01
        5.497246967388161e-01
        5.548592886160301e-01
        5.599728345271531e-01
        5.650651405139762e-01
        5.701360134239264e-01
        5.751852609173937e-01
        5.802126914750259e-01
        5.852181144049929e-01
        5.902013398502205e-01
        5.951621787955911e-01
        6.001004430751128e-01
        6.050159453790576e-01
        6.099084992610651e-01
        6.147779191452151e-01
        6.196240203330663e-01
        6.244466190106619e-01
        6.292455322555019e-01
        6.340205780434813e-01
        6.387715752557943e-01
        6.434983436858043e-01
        6.482007040458788e-01
        6.528784779741906e-01
        6.575314880414823e-01
        6.621595577577967e-01
        6.667625115791711e-01
        6.713401749142954e-01
        6.758923741311349e-01
        6.804189365635158e-01
        6.849196905176745e-01
        6.893944652787704e-01
        6.938430911173605e-01
        6.982653992958381e-01
        7.026612220748322e-01
        7.070303927195706e-01
        7.113727455062038e-01
        7.156881157280909e-01
        7.199763397020476e-01
        7.242372547745537e-01
        7.284706993279236e-01
        7.326765127864361e-01
        7.368545356224248e-01
        7.410046093623291e-01
        7.451265765927062e-01
        7.492202809662000e-01
        7.532855672074728e-01
        7.573222811190946e-01
        7.613302695873916e-01
        7.653093805882540e-01
        7.692594631929024e-01
        7.731803675736124e-01
        7.770719450093977e-01
        7.809340478916512e-01
        7.847665297297438e-01
        7.885692451565806e-01
        7.923420499341149e-01
        7.960848009588196e-01
        7.997973562671141e-01
        8.034795750407502e-01
        8.071313176121525e-01
        8.107524454697167e-01
        8.143428212630628e-01
        8.179023088082451e-01
        8.214307730929177e-01
        8.249280802814557e-01
        8.283940977200314e-01
        8.318286939416457e-01
        8.352317386711152e-01
        8.386031028300134e-01
        8.419426585415662e-01
        8.452502791355033e-01
        8.485258391528615e-01
        8.517692143507448e-01
        8.549802817070359e-01
        8.581589194250628e-01
        8.613050069382191e-01
        8.644184249145361e-01
        8.674990552612101e-01
        8.705467811290808e-01
        8.735614869170644e-01
        8.765430582765373e-01
        8.794913821156746e-01
        8.824063466037383e-01
        8.852878411753202e-01
        8.881357565345354e-01
        8.909499846591674e-01
        8.937304188047661e-01
        8.964769535086964e-01
        8.991894845941382e-01
        9.018679091740380e-01
        9.045121256550118e-01
        9.071220337411980e-01
        9.096975344380619e-01
        9.122385300561506e-01
        9.147449242147984e-01
        9.172166218457825e-01
        9.196535291969289e-01
        9.220555538356686e-01
        9.244226046525438e-01
        9.267545918646627e-01
        9.290514270191066e-01
        9.313130229962833e-01
        9.335392940132328e-01
        9.357301556268803e-01
        9.378855247372397e-01
        9.400053195905654e-01
        9.420894597824527e-01
        9.441378662608887e-01
        9.461504613292497e-01
        9.481271686492487e-01
        9.500679132438312e-01
        9.519726215000186e-01
        9.538412211717003e-01
        9.556736413823750e-01
        9.574698126278376e-01
        9.592296667788168e-01
        9.609531370835583e-01
        9.626401581703575e-01
        9.642906660500387e-01
        9.659045981183817e-01
        9.674818931584975e-01
        9.690224913431497e-01
        9.705263342370234e-01
        9.719933647989422e-01
        9.734235273840317e-01
        9.748167677458299e-01
        9.761730330383451e-01
        9.774922718180602e-01
        9.787744340458836e-01
        9.800194710890480e-01
        9.812273357229544e-01
        9.823979821329635e-01
        9.835313659161335e-01
        9.846274440829048e-01
        9.856861750587294e-01
        9.867075186856491e-01
        9.876914362238178e-01
        9.886378903529718e-01
        9.895468451738442e-01
        9.904182662095278e-01
        9.912521204067818e-01
        9.920483761372862e-01
        9.928070031988414e-01
        9.935279728165133e-01
        9.942112576437249e-01
        9.948568317632942e-01
        9.954646706884165e-01
        9.960347513635933e-01
        9.965670521655072e-01
        9.970615529038419e-01
        9.975182348220475e-01
        9.979370805980530e-01
        9.983180743449220e-01
        9.986612016114569e-01
        9.989664493827454e-01
        9.992338060806554e-01
        9.994632615642732e-01
        9.996436951481997e-01
        9.997805185994304e-01
        9.998791453391053e-01
        9.999449902371761e-01
        9.999834694070191e-01
        9.999999999999999e-01
        """

        return np.array(ys.strip().split('\n'), dtype = np.float64)

    def calculate_barycentric_weights(self):
        """
        Precompute barycentric interpolation weights for all grid points.

        For each grid index in 'self.y_values', this method computes the
        corresponding first-form barycentric weights, and caches them for
        later reuse during interpolation.

        Returns
        -------
        dict
            Dictionary mapping '(j_s, j_o)' to a NumPy array of barycentric
            weights, where:
            - 'j_s' : int
                Starting index of the stencil in `self.y_values`
            - 'j_o' : int
                Offset value
            - 'weights' : np.ndarray
                Barycentric weights for the corresponding stencil points.
        """
        barycentric_weights = {}
        for i in range(len(self.y_values)):
            (j_s,j_e,j_o) = self.find_stencil_endpoints(i)
            stencil_points = self.y_values[j_s:j_e+1]
            weights = np.ones(len(stencil_points), dtype=np.float64)

            for i in range(1,len(stencil_points)):
                for j in range(i):
                    weights[j] *= stencil_points[j] - stencil_points[i]
                    weights[i] *= stencil_points[i] - stencil_points[j]

            weights = 1.0/weights
            barycentric_weights[(j_s,j_o)] = weights

        return barycentric_weights

    def find_stencil_bounds(self, y_interpolate):
        """
        This method performs a binary search on 'self.y_values' to find an
        index 'n' such that 'y_interpolate' lies between neighboring grid
        points. The returned index is later used to determine stencil
        bounds for interpolation.

        The search domain is split into two regions to handle asymmetry
        around the grid center:
        - If 'y_interpolate <= 0', the search is restricted to the lower
        half of the grid.
        - If 'y_interpolate > 0', the search is restricted to the upper
        half of the grid.

        Parameters
        ----------
        y_interpolate : float
            The coordinate at which interpolation is requested.

        Returns
        -------
        int
            Index 'n' such that 'y_interpolate' lies between adjacent grid
            points:
            - For 'y_interpolate <= 0':
                self.y_values[n] <= y_interpolate < self.y_values[n + 1]
            - For 'y_interpolate > 0':
                self.y_values[n - 1] < y_interpolate <= self.y_values[n]

            Returns '-1' if no valid index is found.
        """
        Ny = len(self.y_values)
        
        if y_interpolate <= 0:
            low, high = 0, int(Ny/2) - 1
            while low <= high:

                mid = (low + high) // 2
                if self.y_values[mid] <= y_interpolate < self.y_values[mid + 1]:
                    return mid
                elif y_interpolate < self.y_values[mid]:
                    high = mid - 1
                else:
                    low = mid + 1

            return -1 
        else:
            low, high = int(Ny/2), Ny - 1
            while low <= high:
                mid = (low + high) // 2

                if self.y_values[mid - 1] < y_interpolate <= self.y_values[mid]:
                    return mid
                elif y_interpolate > self.y_values[mid]:
                    low = mid + 1
                else:
                    high = mid - 1
                    
            return -1 
        
    def find_stencil_endpoints(self, n):   
        """
        Given a reference index `n` in grid points, this method computes
        the start index 'j_s', end index 'j_e', and origin offset 'j_o'
        defining a contiguous stencil of length 'lagrange_order' (q).

        The stencil is centered about 'n' when possible. Near domain
        boundaries, the stencil is shifted to remain within valid grid
        bounds, and the origin offset 'j_o' records the displacement of
        the reference index within the stencil.

        Parameters
        ----------
        n : int
            Reference grid index returned by the find_stencil_bounds function.

        Returns
        -------
        tuple of int
            (j_s, j_e, j_o) where:
            - 'j_s' : int
                Starting index of the stencil in grid points.
            - 'j_e' : int
                Ending index of the stencil in grid points.
            - 'j_o' : int
                Offset of the reference index within the stencil, such that
                the reference point corresponds to stencil index 'j_o'.
        """  
        j_s = None
        j_e = None
        j_o = None

        if n <= ((len(self.y_values)/2) - 1):
            j_o = max(math.ceil(self.lagrange_order/2) - n - 1, 0)
            j_s = n - math.ceil(self.lagrange_order/2) + 1 + j_o
        else:
            j_o = min(len(self.y_values) - n - math.ceil(self.lagrange_order/2), 0)
            j_s = n - math.floor(self.lagrange_order/2) + j_o

        j_e = j_s + self.lagrange_order - 1

        return (j_s,j_e,j_o)

    def calculate_interpolation_weights(self, stencil_points, barycentric_weights, y_interpolate):
        """
        Given a set of stencil grid points and their precomputed first-form
        barycentric weights, this function computes the interpolation weights.
        These weights can be applied to function values at the stencil points to
        obtain the interpolated value at 'y_interpolate'.

        The interpolation weights λ_j are given by:

            λ_j(y) = (w_j / (y - x_j)) / Σ_k (w_k / (y - x_k))

        where:
            - x_j are the stencil grid points
            - w_j are the barycentric weights

        Parameters
        ----------
        stencil_points : np.ndarray
            One-dimensional array of grid points forming the interpolation
            stencil.
        barycentric_weights : np.ndarray
            First-form barycentric weights corresponding to `stencil_points`.
        y_interpolate : float
            The coordinate at which the interpolant is evaluated.

        Returns
        -------
        np.ndarray
            Array of interpolation weights of the same length as 'stencil_points'. 
        """

        differences = y_interpolate - stencil_points

        numerators = np.divide(
            barycentric_weights,
            differences,
            out=np.sign(barycentric_weights) * np.finfo(np.float64).max,
            where=differences != 0
        )
        
        denominator = np.sum(numerators)

        weights = numerators / denominator

        return weights
    
    def get_stencil_weights(self, y_interpolate):
        n = self.find_stencil_bounds(y_interpolate)

        j_s, j_e, j_o = self.find_stencil_endpoints(n)

        stencil_points = self.y_values[j_s : j_e + 1]
        
        barycentric_weights = self.barycentric_weights[(j_s, j_o)]

        return self.calculate_interpolation_weights(stencil_points, barycentric_weights, y_interpolate)
