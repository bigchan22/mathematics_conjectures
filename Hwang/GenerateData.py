import generate_input as gen

DIR_PATH = '/Data/Min/mathematics_conjectures/Hwang/Data/'
!mkdir {DIR_PATH}
gen.generate_data(DIR_PATH=DIR_PATH, N=9, connected=True)