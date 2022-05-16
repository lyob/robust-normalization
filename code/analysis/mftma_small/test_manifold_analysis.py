import os
from scipy import io as sio
import numpy as np

from manifold_analysis import *


class VariableLoader:
    def __init__(self, data_dir, suffix):
        self.data_dir = data_dir
        self.suffix = suffix

    def load_mat(self, filename, dataname):
        all_data = sio.loadmat(filename)
        data = all_data[dataname]
        return data

    def load_var(self, var_name):
        filename = var_name + self.suffix
        return self.load_mat(os.path.join(self.data_dir, filename), var_name)


def test_maxproj():
    data_dir = os.path.join('data', 'analysis', 'maxproj')
    loader = VariableLoader(data_dir, '_maxproj.mat')

    # Load the test inputs
    t_vec = loader.load_var('t_vec')
    sD1 = loader.load_var('sD1')
    sc = loader.load_var('sc')
    # Load the test outputs
    ss = loader.load_var('ss')
    gg = loader.load_var('gg')

    # Run the function on test inputs
    ss_output, gg_output = maxproj(t_vec, sD1, sc)

    # Check the outputs against test outputs
    assert np.allclose(ss, ss_output)
    assert np.allclose(gg, gg_output)


def test_minimize_vt_sq():
    data_dir = os.path.join('data', 'analysis', 'minimize_vt_sq')
    loader = VariableLoader(data_dir, '_minimize_vt_sq.mat')

    # Load the test inputs
    tt = loader.load_var('tt')
    Tk = loader.load_var('Tk')
    kappa = loader.load_var('kappa')[0, 0]

    # Load the test outputs
    v_k = loader.load_var('v_k')
    vt_k = loader.load_var('vt_k')
    exitflag = loader.load_var('exitflag')
    alphak = loader.load_var('alphak')
    vminustsqk = loader.load_var('vminustsqk')

    # Run the function on test inputs
    v_k_out, vt_k_out, exitflag_out, alphak_out, vminustsqk_out = minimize_vt_sq(tt, Tk, kappa)

    # Check the outputs against the test outputs
    assert np.allclose(v_k_out, v_k)
    assert np.allclose(vt_k_out, vt_k)
    assert np.allclose(alphak_out, alphak)
    assert np.allclose(vminustsqk_out, vminustsqk)


def test_each_manifold_analysis_D1():
    data_dir = os.path.join('data', 'analysis', 'each_manifold_analysis_D1')
    loader = VariableLoader(data_dir, '_each_manifold_analysis_D1')

    # Load the test inputs
    sD1 = loader.load_var('sD1')
    kappa = loader.load_var('kappa')[0, 0]
    n_t = loader.load_var('n_t')[0, 0]
    t_vec = loader.load_var('t_vec')

    # Load the test outputs
    a_Mfull = loader.load_var('a_Mfull')
    R_M = loader.load_var('R_M')
    D_M = loader.load_var('D_M')

    # Run the function on the test inputs
    a_Mfull_out, R_M_out, D_M_out = each_manifold_analysis_D1(sD1, kappa, n_t, t_vec=t_vec)

    # Check the outputs against the test cases
    assert np.allclose(a_Mfull_out, a_Mfull)
    assert np.allclose(R_M_out, R_M)
    assert np.allclose(D_M_out, D_M)


def test_manifold_analysis():
    data_dir = os.path.join('data', 'analysis', 'manifold_analysis')
    loader = VariableLoader(data_dir, '_manifold_analysis.mat')

    # Load the test inputs
    XtotT = loader.load_var('XtotT')
    XtotT = [XtotT[i][0] for i in range(XtotT.shape[0])]
    kappa = loader.load_var('kappa')[0, 0]
    n_t = loader.load_var('n_t')[0, 0]
    # Load the fixed t vectors
    t_vec_all = loader.load_var('t_vec_all')
    t_vec_all = [t_vec_all[0][i] for i in range(t_vec_all.shape[1])]

    # Load the test outputs
    a_Mfull_vec = loader.load_var('a_Mfull_vec')
    R_M_vec = loader.load_var('R_M_vec')
    D_M_vec = loader.load_var('D_M_vec')

    # Run the function on the test inputs
    a_Mfull_vec_out, R_M_vec_out, D_M_vec_out = manifold_analysis(XtotT, kappa, n_t, t_vecs=t_vec_all)

    # Check the outputs against the test cases
    assert np.allclose(a_Mfull_vec_out, a_Mfull_vec)
    assert np.allclose(R_M_vec_out, R_M_vec)
    assert np.allclose(D_M_vec_out, D_M_vec)


if __name__ == '__main__':
    test_maxproj()
    test_minimize_vt_sq()
    test_each_manifold_analysis_D1()
    test_manifold_analysis()
