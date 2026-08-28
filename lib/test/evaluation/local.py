from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.prj_dir = 'C:/UNTrack'  # Base directory for saving network checkpoints.
    settings.musthsi_path = 'C:/UNTrack/data'
    settings.network_path = 'C:/UNTrack/lib/test/networks'    # Where tracking networks are stored.
    settings.result_plot_path = 'C:/UNTrack/lib/test/result_plots'
    settings.results_path = 'C:/UNTrack/output'    # Where to store tracking results
    settings.segmentation_path = 'C:/UNTrack/lib/test/segmentation_results'
    settings.save_dir = 'C:/UNTrack/output'

    return settings

