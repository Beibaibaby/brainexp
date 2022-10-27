import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
#x_test_plot=np.load('./plot3/x_test_plot.npy')
#final_output_test=np.load('./plot3/final_output_test.npy')


x_test_plot=np.load('./plot3/y_test_plot.npy')
final_output_test=np.load('./plot3/final_output_test_2.npy')

np.save('x_test_plot',x_test_plot)
np.save('final_output_test',final_output_test)
plt.scatter(x_test_plot,final_output_test,alpha=1,s=100)
lims = [
    #np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    0,
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
#fig.savefig('reconstruction.png')
plt.show()