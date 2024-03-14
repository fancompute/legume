import unittest
import legume 
import numpy as np
import scipy.constants as cs


class TestHOP(unittest.TestCase):
    '''
    Test of a multilayered structure with active
    quantum wells. It is compared with legume 
    data.
    '''
    def test_pol(self):

        g_max = 4.1
        # ------------- structure 
        eps_GaAs = 3.56**2
        eps_AlGaAs = 3.32**2
        eps_SiO2 = 1.45**2
        a = 2.45*10**-7
        R= 0.26
        dqw = 2*1e-8/a

        d_etched= 0.3
        d_unetch = 0.93

        M = cs.m_e*0.18
        E_x = 1.53
        osc_str = 10*10**16*np.array([1,1,0])
        loss = 10*1e-6
        # ------------- structure 

        num_k = 12
        lim_k=0.05


        lattice = legume.Lattice("square")

        path = lattice.bz_path([[lim_k*np.pi,lim_k*np.pi], "g",[lim_k*np.pi,0]],[num_k])

        phc = legume.PhotCryst(lattice,eps_l=eps_SiO2)
        phc.add_layer(d=(d_unetch-dqw*3)/2, eps_b=eps_GaAs )
        phc.add_layer(d=dqw, eps_b=eps_AlGaAs )
        phc.add_layer(d=dqw, eps_b=eps_GaAs )
        phc.add_layer(d=dqw, eps_b=eps_AlGaAs )
        phc.add_layer(d=(d_unetch-dqw*3)/2, eps_b=eps_GaAs )
        phc.add_layer(d=d_etched, eps_b=eps_GaAs )
        phc.add_shape(legume.Circle(r=R, eps=1, x_cent=0., y_cent=0.),)


        phc.add_qw(z=phc.layers[3].z_mid,a=a,M=M,E0=E_x,V_shapes=1,loss =loss,osc_str=osc_str)


        pol = legume.HopfieldPol(phc,g_max,truncate_g='abs')
        gme_options = {'gmode_inds':[0,1,2, 3,4,5,6],
                        'numeig':6,
                        'verbose':False,
                        'angles' : path['angles'],
                         'kz_symmetry' : 'odd',
                         'symm_thr': 1e-8,
                         'use_sparse':True}

        exc_options = {'numeig_ex':8,
                'verbose_ex':False}
                
        pol.run(kpoints=path['kpoints'],gme_options=gme_options,exc_options=exc_options)
        
        en = np.load('./tests/data/Polariton_en.npy')
        im = np.load('./tests/data/Polariton_im.npy')
        fr = np.load('./tests/data/Polariton_fr.npy')

        diff_E = np.sum(np.abs(pol.eners-en))
        diff_im = np.sum(np.abs(pol.eners_im-im))
        diff_fr = np.sum(np.abs(pol.fractions_ex-fr))

        self.assertLessEqual(diff_E, 1e-8)
        self.assertLessEqual(diff_im, 1e-8)
        self.assertLessEqual(diff_fr, 1e-8)


if __name__ == '__main__':
    unittest.main()
