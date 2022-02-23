#%%
# install modules
import numpy as np
import matplotlib.pyplot as plt


#%%
# let's start with an array of responses from a population of 10 neurons. 
# this can be thought of as the responses of one layer of a network.
responses = []
num_neurons = 11
responses.append(np.ones(num_neurons))
responses.append(np.ones(num_neurons)/num_neurons);
responses.append(np.ones(num_neurons)); responses[-1][4] = 10
responses.append(np.ones(num_neurons)); responses[-1][4] = 10
responses[-1] = responses[-1] / np.sum(responses[-1])
responses.append(np.ones(num_neurons)); responses[-1][4] = 0.3
responses[-1] = responses[-1] / np.sum(responses[-1])

# print('np.linalg.norm vs sum')
# print(np.linalg.norm(responses[-1]))
# print(np.sum(responses[-1]))

print('inputs')
print(*responses, sep='\n')

# we now define the normalization procedure
def normalize(inputs, sigma=1, exponent=1):
    outputs = []
    for j in range(len(inputs)):
        pool = np.sum(np.square(inputs)) - inputs[j]**2
        outputs.append(inputs[j]**exponent/(sigma**exponent + pool))
    return outputs


normed_responses = []
for j in range(len(responses)):
    normed_responses.append(normalize(responses[j]))

normed_exp2 = []
for j in range(len(responses)):
    normed_exp2.append(normalize(responses[j], exponent=2))

# print('normed responses')
# print(*normed_responses, sep='\n')
# print(*normed_exp2, sep='\n')

"""
plot
"""
x_axis = np.arange(num_neurons)
fig, axes = plt.subplots(len(responses),3, figsize=(10,10), sharey='row', sharex='col')
axes[0,0].set(title="unnormalized")
axes[0,1].set(title="normed responses")
axes[0,2].set(title="normed with exp=2")

for row in range(len(responses)):
    axes[row,0].plot(x_axis, responses[row])
    axes[row,1].plot(x_axis, normed_responses[row])
    axes[row,2].plot(x_axis, normed_exp2[row])

# # all neurons respond the same
# axes[0,0].set(ylabel="all have same response")
# # same as above but scaled to sum to 1
# axes[1,0].set(ylabel="same as above but rescaled")
# # one number is much larger than the others
# axes[2,0].set(ylabel="one neuron much larger")
# # same as above, but scaled to sum to 1
# axes[3,0].set(ylabel="same as above but rescaled")
# # one number is smaller than the others
# axes[4,0].set(ylabel="one neuron much smaller")

fig.tight_layout()

"""initial scale matters. I think the inputs should sum to 1 initally, but I have no justification for that.
Then there's the issue of how the normalization behavior changes with the distribution of neuronal responses""" 

# %% interaction between peak and remaining neurons
# let's plot how the difference between a big peak and a suppressive field interact with each other:

def make_sum_to_one(inputs):
    return inputs/inputs.sum()

# scale_values = range(0, 101, 5)
scale_values = np.linspace(0, 2, 20)
# print(f'scale values: \n{scale_values}')

baseline = np.ones(10)
responses = []
responses_exp2 = []

for j in range(len(scale_values)):
    # exp = 1
    baseline[4] = scale_values[j]
    responses.append(make_sum_to_one(baseline))
    responses[-1] = normalize(responses[-1])

    # exp = 2
    baseline[4] = scale_values[j]
    responses_exp2.append(make_sum_to_one(baseline))
    responses_exp2[-1] = normalize(responses_exp2[-1], exponent=2)

scale_differences = [responses[j][4] / responses[j][0] for j in range(len(responses))]
scale_differences_exp2 = [responses_exp2[j][4] / responses_exp2[j][0] for j in range(len(responses_exp2))]
# print(f'scale_differences: \n{scale_differences}')

# plot the ratio between peak and non-peak magnitudes beofre and after norm
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
row = 0
ax[row].plot(scale_values, scale_differences)
ax[row].set(title="exponent value = 1")
row = 1
ax[row].plot(scale_values, scale_differences_exp2)
ax[row].set(title="exponent value = 2", xlabel="scale difference between magnitude of max amp neuron and other neurons")


#%% how does 
# show how norm changes the values of the magnitudes for the peak and suppressive neurons 

peak_differences = [responses[j][4]/normalize(responses[j])[4] for j in range(len(responses))]
peak_differences_exp2 = [normalize(responses_exp2[j])[4]/responses_exp2[j][4] for j in range(len(responses_exp2))]
print(len(peak_differences))

supp_differences = [responses[j][0]/normalize(responses[j])[0] for j in range(len(responses))]
supp_differences_exp2 = [normalize(responses_exp2[j])[0]/responses_exp2[j][0] for j in range(len(responses_exp2))]

fig, ax = plt.subplots(4, 1, figsize=(10,13), sharex=True)
row = 0
ax[row].plot(scale_values, peak_differences)
row = 1
ax[row].plot(scale_values, peak_differences_exp2)
row = 2
ax[row].plot(scale_values, supp_differences)
row = 3
ax[row].plot(scale_values, supp_differences_exp2)


# %% geometric interpretation of the code


