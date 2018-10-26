from init import *

# Show distribution of labels
def show_distributions():
    counts = np.zeros(10)
    label_values = np.asarray(range(1,11))
    for i in range(10):
        counts[i] = sum(labels == i+1)

    plt.figure(1)
    plt.bar(label_values, counts)
    plt.xticks(label_values)
    plt.ylabel('Count')
    plt.xlabel('Label')
    plt.title('Number of occurences in each gender')
    plt.show()

def show_values():
    rythm_range = np.arange(0,168) # Indexes from 0 to 167
    rythm = data[:,rythm_range]
    pitch_range = np.arange(168, 216)
    pitch = data[:, pitch_range]
    timbre_range = np.arange(216, 264)
    timbre = data[:, timbre_range]

    songs = np.asarray([10,11,2200,3659])
    iters = len(songs)

    fig = plt.figure(figsize=(16, 8*len(songs)))

    for i in range(iters):
        plt.subplot(iters,3,i*3+1)
        plt.plot(rythm_range, rythm[songs[i],:])
        plt.title("Label: " + str(labels[songs[i],0]))
        plt.subplot(iters,3,i*3+2)
        plt.plot(pitch_range, pitch[songs[i],:])
        plt.subplot(iters,3,i*3+3)
        plt.plot(timbre_range, timbre[songs[i],:])

    plt.subplots_adjust(hspace = 0.35)
    plt.show()

# show_distributions()
show_values()
