import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)

    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Training...')
    display.display(fig)
    plt.clf()
    plt.title('Training Agent using DRL')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Scores')
    plt.plot(scores, 'g--', label="Scores")
    plt.plot(mean_scores, 'r-o', label="Mean scores")
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)