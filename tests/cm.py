print ('Running cm.py')
import pysasf
from pysasf.basindata import BasinData
import pysasf.clarkeminella as cm
from pysasf import plots
from pysasf import stats

datafile = "data/arvorezinha_database.xlsx"
data = BasinData(datafile)

print(data.infos())
print(data.means())
print(data.std())

data.set_output_folder('output')

data.calculate_and_save_all_proportions(load=False)

combs, Ps = data.load_combs_and_props_from_files(data.output_folder+'/C9E9L20Y24_combs.txt',data.output_folder+'/C9E9L20Y24_props.txt')

print('All combinations indexes:', combs)
print('All proportions calculated:', Ps)

Pfea = cm.cm_feasebles(Ps)
print("The total number of feasible solution is:", len(Pfea))

Pcr = cm.confidence_region(Pfea)
print("The total number of points in 95% confidence region is:", len(Pcr))

plots.draw_hull(Pcr, path = data.output_folder, 
                filename='convex_hull', savefig = False)

for n in [2,4,8,12,16,20,24]:
    combs,Ps = stats.randon_props_subsamples(data, 'Y', n)
    P_feas = cm.cm_feasebles(Ps)
    P_cr = stats.confidence_region(P_feas,space_dist='mahalanobis0')
    name = 'confidence_region_Y'+str(n)
    ax = plots.draw_hull(P_cr, savefig = True, 
                         filename = data.output_folder+name)

cm.run_repetitions_and_reduction (data, 'Y',[2,4,8,12,16,20,24])
cm.run_repetitions_and_reduction (data, 'L',[2,4,8,12,16,20,])

files = [data.output_folder+'/'+'C9E9L20Y24_Y-2-4-8-12-16-20-24.csv',
         arvorezinha.output_folder+'/'+'C9E9L20Y24_L-2-4-8-12-16-20.csv']

plots.plot_cm_outputs(files, 'nSamp', 'CV', savefig=False)
plots.plot_cm_outputs(files, 'nSamp', 'CV', savefig=True)
