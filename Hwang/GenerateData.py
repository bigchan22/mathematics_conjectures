import generate_data as gen

DIR_PATH = '/root/Hwang/mathematics_conjectures/Hwang/Data/'
!mkdir {DIR_PATH}
gen.generate_data(DIR_PATH=DIR_PATH, N=7, primitive = True, connected=False, extended=False, UPTO_N=False)