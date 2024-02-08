import DGM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import norm

DEBUG = False
USE_CHECKPOINT = True

# Option parameters
INTEREST_RATE = 0.05           # Interest rate
SIGMA = 0.25                   # Volatility
STRIKE = 50.                   # Strike
START_TIME, MATURITY_TIME = 0., 1.
INITIAL_SPOT = 50.

# neural network parameters
NUM_LAYERS = 3
NODES_PER_LAYER = 50 if not DEBUG else 5
LEARNING_RATE = 0.001

# Training parameters
SAMPLING_STAGES  = 100 if not DEBUG else 2   # number of times to resample new time-space domain points
STEPS_PER_SAMPLE = 10  if not DEBUG else 2   # number of SGD steps to take before re-sampling
CLIP_GRADIENTS = False

# Sampling parameters
N_INTERIOR_OBS = 1000 if not DEBUG else 5
N_TERMINAL_OBS = 100 if not DEBUG else 2

# Plot options
SPOT_LINSPACE_RESOLUTION = 41  # Points on plot grid for each dimension
# time values at which to examine density
VALUE_TIMES = [0., 1/3*MATURITY_TIME, 2/3*MATURITY_TIME, 1*MATURITY_TIME]

# Save options
MAKE_MODEL_OUTPUT_FIGURE = False
SAVE_MODEL_OUTPUT_FIGURE = True
MAKE_MODEL_LOSS_FIGURE = False
SAVE_MODEL_LOSS_FIGURE = True
MODEL_OUTPUT_FIGURE_NAME = "model_output_BlackScholes_EuropeanCall_moneyness.png" if not DEBUG else "model_output_test.png"
MODEL_LOSS_FIGURE_NAME = "model_loss_BlackScholes_EuropeanCall_moneyness.png" if not DEBUG else "model_loss_test.png"
MODEL_NAME = 'moneyness_model.keras' if not DEBUG else 'debug.keras'

def spot_to_moneyness(spot, strike, continuous_interest_rate, time_until_maturity):
    """ Calculate Moneyness
        m_t = ln(K) - ln(S_t) - r*(T - t)
    """
    return np.log(strike) - np.log(spot) - (continuous_interest_rate) * time_until_maturity

def model_output_to_call_price(model_output, strike, continuous_interest_rate, time_until_maturity):
    """
        C = e^(-r(T-t))*K*(e^(-m_t)N(d_1) - N(d_2))
                          ^^^^^^^^^^^^^^^^^^^^^^^^^
          = e^(-r(T-t))*K*model_output
    """
    return np.exp(-continuous_interest_rate*time_until_maturity)*strike*model_output

def time_to_maturity_to_elapsed_time(time_to_maturity):
    return 1 - time_to_maturity


def sampler(n_interior_obs, n_terminal_obs):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        n_interior_obs: number of space points in the interior of the function's domain to sample 
        n_terminal_obs: number of space points at terminal time to sample (terminal condition)
    ''' 
    if DEBUG:
        print("sample interior")
    interior_shape = [n_interior_obs, 1]
    interior_times = tf.random.uniform(shape=interior_shape, minval=START_TIME, maxval=MATURITY_TIME)

    mu = tf.math.log(INITIAL_SPOT) - tf.math.log(STRIKE) + (INTEREST_RATE - SIGMA**2 / 2) * (MATURITY_TIME - interior_times)
    std_dev = 3 * SIGMA * tf.sqrt(MATURITY_TIME - interior_times)
    moneyness_interior = tf.random.normal(shape=interior_shape, mean=mu, stddev=std_dev)

    ## Don't sample the terminal condition

    # if DEBUG:
    #     print("sample terminal")
    # terminal_shape = [n_terminal_obs, 1]
    # terminal_times = tf.fill(dims=terminal_shape, value=MATURITY_TIME)
    # moneyness_terminal = 3*SIGMA*np.sqrt(terminal_times) * tf.random.uniform(shape=terminal_shape, minval=-1, maxval=1)

    # time = tf.concat([interior_times, terminal_times],axis=0)
    # moneyness = tf.concat([moneyness_interior, moneyness_terminal],axis=0)

    return interior_times, moneyness_interior


def training_loss(model, time_to_maturity, moneyness):
    """
        Lf = Lg + N'(d1)/(I*sqrt(T-t)) LI - sigma^2 exp(-mt) (N'(d1) * (2 d1 - 1)/(2 I))/N(d1) del I / del m
        Lg = - exp(-mt I N'(d1))/(2 sqrt(T-t)) + sigma^2/2 exp(-mt) N'(d1)/(I sqrt(T-t))
        LI = del I / del t + sigma^2/2 (del^2 I / del^2 m) + sigma^2/2 del I / del m
    """
    _, (_, _, Lf) = model_loss_components(model, time_to_maturity, moneyness)
    valid_Lf = tf.boolean_mask(Lf, tf.math.is_finite(Lf))

    differential_loss = tf.reduce_mean(tf.square(valid_Lf))

    desc = f'differential_loss: {differential_loss:.3f}'
    return differential_loss, desc

def model_loss_components(model, time_to_maturity, moneyness):
    I, I_m, I_t, I_mm = createGradients(model, time_to_maturity, moneyness)
    if DEBUG:
        print(f"\nI={I};\nI_m{I_m}'\nI_t={I_t}'\nI_mm={I_mm}\n")

    Lg, LI, Lf = find_lg_li_and_lf(moneyness, time_to_maturity, I, I_t, I_m, I_mm)

    return (I, I_m, I_t, I_mm), (Lg, LI, Lf)

def find_lg_li_and_lf(moneyness, time_to_maturity, I, I_t, I_m, I_mm):
    d1 = calculate_d1(moneyness, time_to_maturity)
    cumulative_probability = norm.cdf(d1)
    probability_density = norm.pdf(d1)

    Lg = lg_loss(moneyness, I, probability_density, time_to_maturity)
    LI = li_loss(I_t, I_mm, I_m)
    Lf = lf_loss(Lg, probability_density, I, time_to_maturity, LI, moneyness, d1, cumulative_probability, I_m)
    return Lg, LI, Lf

def calculate_d1(moneyness, time_to_maturity):
    return (-moneyness + (SIGMA**2 / 2) * (time_to_maturity)) / (SIGMA * tf.sqrt(time_to_maturity))

def lg_loss(moneyness, I, probability_density, time_to_maturity):
    return -tf.exp(-moneyness) * I * probability_density / (2 * tf.sqrt(time_to_maturity)) + (SIGMA**2 / 2) * tf.exp(-moneyness) * probability_density / (I * tf.sqrt(time_to_maturity))

def li_loss(I_t, I_mm, I_m):
    return I_t + (SIGMA**2 / 2) * I_mm + (SIGMA**2 / 2) * I_m

def lf_loss(Lg, probability_density, I, time_to_maturity, LI, moneyness, d1, cumulative_probability, I_m):
    return Lg + probability_density / (I * tf.sqrt(time_to_maturity)) * LI - SIGMA**2 * tf.exp(-moneyness) * probability_density * ((2 * d1 - 1) / (2 * I)) / cumulative_probability * I_m

def createGradients(model, time_to_maturity, moneyness):
    with tf.GradientTape(persistent=True) as outer_tape:
        outer_tape.watch([time_to_maturity, moneyness])
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch([time_to_maturity, moneyness])
            I = model((time_to_maturity, moneyness))
        I_m = inner_tape.gradient(I, moneyness)
        I_t = inner_tape.gradient(I, time_to_maturity)
    I_mm = outer_tape.gradient(I_m, moneyness)

    del inner_tape
    del outer_tape

    return I, I_m, I_t, I_mm

def clip_gradients(gradients):
    if any(tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients):
        print("NaNs in gradients")
    clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
    if any(tf.reduce_any(tf.math.is_nan(grad)) for grad in clipped_gradients):
        print("NaNs in gradients")
    return clipped_gradients

def train(model):
    print('init optimizer')
    # set optimizer
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    # for each sampling stage
    for i in (pbar := tqdm(range(SAMPLING_STAGES))):
        if DEBUG:
            print("sampling...")
            losses = []
        # sample uniformly from the required regions (these form X)
        time, moneyness = sampler(N_INTERIOR_OBS, N_TERMINAL_OBS)
    
        for _ in tqdm(range(STEPS_PER_SAMPLE), leave=False):
            with tf.GradientTape() as tape:
                loss, loss_str_desc = training_loss(model, MATURITY_TIME - time, moneyness)
                if DEBUG:
                    losses.append(loss.numpy().item())
            if DEBUG:
                print('backpropping...')
            gradients = tape.gradient(loss, model.trainable_variables)
            if CLIP_GRADIENTS:
                print('clipping gradients...')
                gradients = clip_gradients(gradients)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if DEBUG:
            print(losses)
        pbar.set_description(loss_str_desc)
    model.save(MODEL_NAME)
    return model

def create_model_output_graph(model):
    print('plotting model output...')
    # LaTeX rendering for text in plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # figure options
    plt.figure()
    plt.figure(figsize = (12,10))

    SPOT_LOW = 0.0 + 1e-10  # spot price lower bound
    SPOT_HIGH = 2*STRIKE    # spot price upper bound
    spot_space = np.linspace(SPOT_LOW, SPOT_HIGH, SPOT_LINSPACE_RESOLUTION).reshape(-1,1)
    for i, current_time in tqdm(enumerate(VALUE_TIMES)):
        time_space = current_time * np.ones_like(spot_space)
        time_to_maturity = tf.convert_to_tensor(MATURITY_TIME - time_space)
        moneyness = tf.convert_to_tensor(spot_to_moneyness(spot_space, STRIKE, INTEREST_RATE, time_to_maturity))

        plt.subplot(2,2,i+1)
        plt.plot(moneyness, model((time_to_maturity, moneyness)), color = 'r', label='Model Output', linewidth = 3, linestyle=':')

        # subplot options
        plt.xlabel(r"Moneyness", fontsize=15, labelpad=10)
        plt.ylabel(r"$I(t,m_t)$", fontsize=15, labelpad=20)
        plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%current_time, fontsize=18, y=1.03)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(linestyle=':')

        if i == 0:
            plt.legend(loc='upper left', prop={'size': 16})

    # adjust space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    if SAVE_MODEL_OUTPUT_FIGURE:
        plt.savefig(MODEL_OUTPUT_FIGURE_NAME)

def create_model_loss_figure(model):
    print('plotting model loss...')
    # LaTeX rendering for text in plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # figure options
    plt.figure()
    plt.figure(figsize = (12,10))

    SPOT_LOW = 0.0 + 1e-10  # spot price lower bound
    SPOT_HIGH = 2*STRIKE    # spot price upper bound
    spot_space = np.linspace(SPOT_LOW, SPOT_HIGH, SPOT_LINSPACE_RESOLUTION).reshape(-1,1)
    for i, current_time in tqdm(enumerate(VALUE_TIMES)):
        time_space = current_time * np.ones_like(spot_space)
        time_to_maturity = tf.convert_to_tensor(MATURITY_TIME - time_space, dtype=tf.float32)
        moneyness = tf.convert_to_tensor(spot_to_moneyness(spot_space, STRIKE, INTEREST_RATE, time_to_maturity), dtype=tf.float32)

        (_, _, _, _), (_, _, Lf) = model_loss_components(model, time_to_maturity, moneyness)

        plt.subplot(2,2,i+1)
        plt.plot(moneyness, Lf, color = 'r', label='Model Loss', linewidth = 3, linestyle=':')
    
        # subplot options
        plt.xlabel(r"Moneyness", fontsize=15, labelpad=10)
        plt.ylabel(r"$Lf$", fontsize=15, labelpad=20)
        plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%current_time, fontsize=18, y=1.03)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(linestyle=':')
    
        if i == 0:
            plt.legend(loc='upper left', prop={'size': 16})
    
    # adjust space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    if SAVE_MODEL_LOSS_FIGURE:
        plt.savefig(MODEL_LOSS_FIGURE_NAME)

def create_loss_graph():
    print('plotting model=SIGMA loss...')
    # LaTeX rendering for text in plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # figure options
    plt.figure()
    plt.figure(figsize = (12,10))

    SPOT_LOW = 0.0 + 1e-10  # spot price lower bound
    SPOT_HIGH = 2*STRIKE    # spot price upper bound
    spot_space = np.linspace(SPOT_LOW, SPOT_HIGH, SPOT_LINSPACE_RESOLUTION).reshape(-1,1)
    for i, current_time in tqdm(enumerate(VALUE_TIMES)):
        time_space = current_time * np.ones_like(spot_space)
        time_to_maturity = tf.convert_to_tensor(MATURITY_TIME - time_space, dtype=tf.float32)
        moneyness = tf.convert_to_tensor(spot_to_moneyness(spot_space, STRIKE, INTEREST_RATE, time_to_maturity), dtype=tf.float32)

        (_, _, Lf) = find_lg_li_and_lf(moneyness, time_to_maturity, SIGMA, 0, 0, 0)

        plt.subplot(2,2,i+1)
        plt.plot(moneyness, Lf, color = 'r', label='Model=SIGMA Loss', linewidth = 3, linestyle=':')
    
        # subplot options
        plt.xlabel(r"Moneyness", fontsize=15, labelpad=10)
        plt.ylabel(r"$Lf$", fontsize=15, labelpad=20)
        plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%current_time, fontsize=18, y=1.03)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(linestyle=':')
    
        if i == 0:
            plt.legend(loc='upper left', prop={'size': 16})
    
    # adjust space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    plt.savefig("model=SIGMA_loss_figure.png")

def main():
    if USE_CHECKPOINT and os.path.exists(MODEL_NAME):
        print("loading trained model...")
        model = tf.keras.models.load_model(MODEL_NAME)
    else:
        print('init model')
        # model = DGM.DGMNet(NODES_PER_LAYER, NUM_LAYERS, 1)
        model = DGM.TrivialNet(1, 1)
        train(model)

    if MAKE_MODEL_LOSS_FIGURE:
        create_model_loss_figure(model)

    if MAKE_MODEL_OUTPUT_FIGURE:
        create_model_output_graph(model)

    create_loss_graph()

if __name__ == '__main__':
    main()
