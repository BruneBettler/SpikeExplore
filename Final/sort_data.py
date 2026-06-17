from extract_data import getXMLData, get_prm_info
import spikeinterface as si
import numpy as np
from probeinterface import Probe
import spikeinterface.preprocessing as spre
from spikeinterface.sorters import run_sorter

def create_recording(dat_path, xml_path, siteLoc, uV_per_count=0.195, shank=2):
    channel_ids, skippedChannels, xml_data = getXMLData(xml_path)
    nBits, nChannels, samplingRate, offset, lfpSamplingRate = xml_data
    
    total_contact_sites = len(channel_ids)
    
    single_shank_contact_site_num = int(total_contact_sites / 2)
    
    full_recording = si.read_binary(file_paths=dat_path, sampling_frequency=samplingRate, dtype=f"int{nBits}", num_channels=nChannels, channel_ids=np.arange(nChannels), is_filtered=False, gain_to_uV=uV_per_count, offset_to_uV=0.0)
    shank2_channel_ids = np.array(channel_ids[32:])
    shank2_recording = full_recording.select_channels(channel_ids=shank2_channel_ids)

    # single shank probe 
    probe_m_x = siteLoc[:32,0] #- np.min(siteLoc[32:,0]) # shift x coordinates so the minimum is at 0
    probe_m_y = siteLoc[:32,1] #- np.min(siteLoc[32:,1]) # shift y coordinates so the minimum is at 0

    probe_m = Probe(ndim=2, si_units='um')
    probe_m.set_contacts(positions=np.column_stack((probe_m_x, probe_m_y)), shapes=np.array(['square'] * single_shank_contact_site_num), shape_params={"width": 13})
    
    # set the contact IDs for each site 
    probe_m.set_device_channel_indices(np.arange(single_shank_contact_site_num))
    probe_m.set_contact_ids(shank2_channel_ids)
    probe_m.create_auto_shape()
    
    # link the probe geometry to our recording object 
    shank2_probe_m = shank2_recording.set_probe(probe_m, in_place=False)
    shank2_skipped_channels = np.intersect1d(shank2_channel_ids, skippedChannels)

    m_rec_clean = shank2_probe_m.remove_channels(shank2_skipped_channels)
    
    return m_rec_clean
    
    
def preProcess_recording(rec): 
    P1 = spre.center(rec, mode='median', dtype='float32')
    P2 = spre.highpass_filter(P1, freq_min=300.0) # TODO: consider bandpass 
    #P3_glob = spre.common_reference(recording = P2, reference = 'global', operator='median') 
    P3_loc = spre.common_reference(recording = P2, reference='local', operator='median', local_radius=(40, 180), min_local_neighbors=5)
    #P4_z = spre.zscore(P3_loc, mode="median+mad")
    P4_loc = spre.whiten(P3_loc, mode="local", radius_um=100) 
    
    return P4_loc


# Kilosort 4
def runKilosort4(rec, save_path):
    params = {
        'batch_size': 80000,    #  'batch_size': 'Number of samples included in each batch of data. Default value: 60000 Min, max: (1, inf) Type: int'
        'nblocks': 0,           #   'nblocks': 'Number of non-overlapping blocks for drift correction (additional nblocks-1 blocks are created in the overlaps). Default value: 1 Min, max: (0, inf) Type: int', 
        'Th_universal': 9,      #   'Th_universal': 'Spike detection threshold for universal templates. Th(1) in previous versions of Kilosort. Default value: 9 Min, max: (0, inf) Type: float', 
        'Th_learned': 8,        #   'Th_learned': 'Spike detection threshold for learned templates. Th(2) in previous versions of Kilosort. Default value: 8 Min, max: (0, inf) Type: float', 
        'nt': 61,               #   'nt': 'Number of samples per waveform and the size of symmetric padding for filtering. Note that `nt` must be an odd number. Default value: 61 Min, max: (1, inf) Type: int', 
        'shift': None,          #   'shift': 'Scalar shift to apply to data before all other operations. In most cases this should be left as None, but may be necessary for float32 data for example. If needed, `shift` and `scale` should be set such that data is roughly in the range -100 to +100. If set, data will be `data = data*scale + shift`. Default value: None Min, max: (-inf, inf) Type: float', 
        'scale': 0.195,          # 'scale': 'Scaling factor to apply to data before all other operations. In most cases this should be left as None, but may be necessary for float32 data for example. If needed, `shift` and `scale` should be set such that data is roughly in the range -100 to +100. If set, data will be `data = data*scale + shift`. Default value: None Min, max: (-inf, inf) Type: float', 
        'batch_downsampling': 1, #  'batch_downsampling': 'Number of batches skipped for each batch used for sorting. For example, if `batch_downsampling = 10`, then only every 10th batch will be used. In general, this should be left as the default (using all batches). Default value: 1 Min, max: (1, inf) Type: int', 
        'artifact_threshold': np.inf, #   'artifact_threshold': 'If a batch contains absolute values above this number, it will be zeroed out under the assumption that a recording artifact is present. By default, the threshold is infinite (so that no zeroing occurs). Default value: inf Min, max: (0, inf) Type: float', 
        'nskip': 25, #   'nskip': 'Batch stride for computing whitening matrix. Default value: 25 Min, max: (1, inf) Type: int', 
        'whitening_range': 20,  #   'whitening_range': 'Number of nearby channels used to estimate the whitening matrix. Default value: 32 Min, max: (1, inf) Type: int', 
        'highpass_cutoff': 300, #   'highpass_cutoff': 'Critical frequency for highpass Butterworth filter applied to data. Default value: 300 Min, max: (0, inf) Type: float',
        'binning_depth': 5, #   'binning_depth': 'For drift correction, vertical bin size in microns used for 2D histogram. Default value: 5 Min, max: (0, inf) Type: float', 
        'sig_interp': 20, #   'sig_interp': 'Approximate spatial smoothness scale in units of microns. Default value: 20 Min, max: (0, inf) Type: float', 
        'drift_smoothing': [0.5, 0.5, 0.5], #   'drift_smoothing': 'Amount of gaussian smoothing to apply to the spatiotemporal drift estimation, for correlation, time (units of registration blocks), and y (units of batches) axes. The y smoothing has no effect for `nblocks = 1`. Adjusting smoothing for the correlation axis is not recommended. Default value: [0.5, 0.5, 0.5] Min, max: (None, None) Type: list', 
        'nt0min': None, #   'nt0min': "Sample index for aligning waveforms, so that their minimum or maximum value happens here. Defaults to `int(20 * settings['nt']/61)`. Default value: None Min, max: (0, inf) Type: int", 
        'dmin': 8, #   'dmin': 'Vertical spacing of template centers used for spike detection, in microns. Determined automatically by default. Default value: None Min, max: (0, inf) Type: float',  
        'dminx': 5, #   'dminx': 'Horizontal spacing of template centers used for spike detection, in microns. The default 32um should work well for Neuropixels 1 and Neuropixels 2 probes. For other probe geometries, try setting this to the median lateral distance between contacts to start. Default value: 32 Min, max: (0, inf) Type: float', 
        'min_template_size': 10, #   'min_template_size': 'Standard deviation of the smallest, spatial envelope Gaussian used for universal templates. Default value: 10 Min, max: (0, inf) Type: float'
        'template_sizes': 5, #   'template_sizes': 'Number of sizes for universal spike templates (multiples of the min_template_size). Default value: 5 Min, max: (1, inf) Type: int', 
        'nearest_chans': 5,   # 'nearest_chans': 'Number of nearest channels to consider when finding local maxima during spike detection. Default value: 10 Min, max: (1, inf) Type: int', 
        'nearest_templates': 100,  # 'nearest_templates': 'Number of nearest spike template locations to consider when finding local maxima during spike detection. Default value: 100 Min, max: (1, inf) Type: int', 
        'max_channel_distance': 10, #   'max_channel_distance': 'Templates farther away than this from their nearest channel will not be used. Also limits distance between compared channels during clustering. Default value: 32 Min, max: (1, inf) Type: float',
        'max_peels': 100, #   'max_peels': 'Number of iterations to do over each batch of data in the matching pursuit step. More iterations may detect more overlapping spikes. Default value: 100 Min, max: (1, 10000) Type: int', 
        'templates_from_data': False, #   'templates_from_data': 'Indicates whether spike shapes used in universal templates should be estimated from the data or loaded from the predefined templates. Default value: True Min, max: (None, None) Type: bool', 
        'n_templates': 6, #   'n_templates': 'Number of single-channel templates to use for the universal templates (only used if templates_from_data is True). Default value: 6 Min, max: (1, inf) Type: int', 
        'n_pcs': 6, #   'n_pcs': 'Number of single-channel PCs to use for extracting spike features (only used if templates_from_data is True). Default value: 6 Min, max: (1, inf) Type: int', 
        'Th_single_ch': 6, #   'Th_single_ch': 'For single channel threshold crossings to compute universal- templates. In units of whitened data standard deviations. Default value: 6 Min, max: (0, inf) Type: float', 
        'acg_threshold': 0.2, #   'acg_threshold': 'Fraction of refractory period violations that are allowed in the ACG compared to baseline; used to assign "good" units. Default value: 0.2 Min, max: (0, inf) Type: float', 
        'ccg_threshold': 0.25, #   'ccg_threshold': 'Fraction of refractory period violations that are allowed in the CCG compared to baseline; used to perform splits and merges. Default value: 0.25 Min, max: (0, inf) Type: float', 
        'cluster_neighbors': 10, #   'cluster_neighbors': 'Number of nearest spike neighbors to search for in `clustering_qr.neigh_mat` when building the adjacency matrix that defines the graph for clustering. Note that changes to this parameter will affect resource usage and sorting time. Default value: 10 Min, max: (2, inf) Type: int', 
        'cluster_downsampling': 20, #   'cluster_downsampling': 'Inverse fraction of spikes used as landmarks during clustering. By default, all spikes are used up to a maximum of `max_cluster_subset=25000`. The old default behavior (version < 4.1.0) is equivalent to `max_cluster_subset=None, cluster_downsampling=20`. Versions 4.1.0 through 4.1.2 defaulted to `max_cluster_subset=25000, cluster_downsampling=1`. The default value was reverted to 20 in version 4.1.3 because many users reported memory issues with the new default. We have found that values betwen 5-20 also work well in most cases. Default value: 20 Min, max: (1, inf) Type: int', 
        'max_cluster_subset': 25000, #   'max_cluster_subset': 'Maximum number of spikes to use when searching for nearest neighbors to build the graph used for clustering. Within each clustering center, only a subset of spikes is searched with the size determined by `cluster_downsampling` and the total number of spikes. This sets a maximum on the size of that subset, so that it will not grow without bound for very long recordings. Using a very large number of spikes is not necessary and causes performance bottlenecks. Use `max_cluster_subset = None` if you do not want a limit on the subset size. The old default behavior (version < 4.1.0) is equivalent to `max_cluster_subset=None, cluster_downsampling=20`. Versions 4.1.0 through 4.1.2 defaulted to `max_cluster_subset=25000, cluster_downsampling=1`. Note: In practice, the actual number of spikes used may increase or decrease slightly while staying under the maximum. This happens because the maximum is set by adjusting `cluster_downsampling` on the fly so that it results in a set no larger than the given size. Default value: 25000 Min, max: (1, inf) Type: int', 
        'x_centers': 2,  #   'x_centers': 'Number of x-positions to use when determining center points for template groupings. If None, this will be determined automatically by finding peaks in channel density. For 2D array type probes, we recommend specifying this so that centers are placed every few hundred microns. Default value: None Min, max: (1, inf) Type: int', 
        'cluster_init_seed': 5, #   'cluster_init_seed': 'Random seed for kmeans++ algorithm used to initialize the graph for clustering. Default value: 5 Min, max: (1, inf) Type: int', 
        'duplicate_spike_ms': 0.25, #   'duplicate_spike_ms': 'Time in ms for which subsequent spikes from the same cluster are assumed to be artifacts. A value of 0 disables this step. NOTE: this was formerly handled by `duplicate_spike_bins`, which has been deprecated. The new default of 0.25ms is equivalent to the old default of 7 bins for a 30kHz sampling rate. Default value: 0.25 Min, max: (0, inf) Type: float', 
        'position_limit': 100, #   'position_limit': 'Maximum distance (in microns) between channels that can be used to estimate spike positions in `postprocessing.compute_spike_positions`. This does not affect spike sorting, only how positions are estimated after sorting is complete. Default value: 100 Min, max: (0, inf) Type: float', 
        'do_CAR': True, #   'do_CAR': 'If True, common average reference is performed. Default is True. (run_kilosrt parameter)', 
        'invert_sign': False, #   'invert_sign': 'Invert the sign of the data. Default value: False. (run_kilosort parameter)', 
        'save_extra_vars': False, #   'save_extra_vars': 'If True, additional kwargs are saved to the output. Default is False. (run_kilosort parameter)', 
        'save_preprocessed_copy': False, #   'save_preprocessed_copy': 'Save a pre-processed copy of the data (including drift correction) to temp_wh.dat in the results directory and format Phy output to use that copy of the data. (run_kilosort parameter)', 
        'torch_device': 'cuda', #   'torch_device': "Select the torch device auto/cuda/cpu. Default is 'auto'. (run_kilosort parameter)", 
        'bad_channels': None, #   'bad_channels': 'A list of channel indices (rows in the binary file) that should not be included in sorting. Listing channels here is equivalent to excluding them from the probe dictionary. (run_kilosort parameter)',
        'clear_cache': False, #   'clear_cache': 'If True, force pytorch to free up memory reserved for its cache in between memory-intensive operations. Note that setting `clear_cache=True` is NOT recommended unless you encounter GPU out-of-memory errors, since this can result in slower sorting. (run_kilosort parameter)', 
        'do_correction': False, #   'do_correction': 'If True, drift correction is performed. Default is True. (spikeinterface parameter)', 
        'skip_kilosort_preprocessing': False, #   'skip_kilosort_preprocessing': 'Can optionally skip the internal kilosort preprocessing. (spikeinterface parameter)', 
        'keep_good_only': False, #   'keep_good_only': "If True, only the units labeled as 'good' by Kilosort are returned in the output. (spikeinterface parameter)", 
        'use_binary_file': True, #   'use_binary_file': 'If True then Kilosort is run using a binary file. In this case, if the input recording is not binary compatible, it is written to a binary file in the output folder. If False then Kilosort is run on the recording object directly using the RecordingExtractorAsArray object. If None, then if the recording is binary compatible, the sorter will use the binary file, otherwise the RecordingExtractorAsArray. Default is True. (spikeinterface parameter)', 
        'delete_recording_dat': True,   # 'delete_recording_dat': 'If True, if a temporary binary file is created, it is deleted after the sorting is done. Default is True. (spikeinterface parameter)',
        'pool_engine': 'process',  #   'n_jobs': 'Number of jobs (when saving to binary) - default global',
        'n_jobs': 1, 
        'chunk_duration': '1s',  #   'chunk_duration': "Chunk duration in s if float or with units if str (e.g. '1s', '500ms') (when saving to binary) - default global"
        'progress_bar': True, #   'progress_bar': 'If True, progress bar is shown (when saving to binary) - default global'
        'mp_context': None, 
        'max_threads_per_worker': 1}  
        #'total_memory': "Total memory usage (e.g. '500M', '2G') (when saving to binary) - default global", 
        #'chunk_size': 'Number of samples per chunk (when saving to binary) - default global'
        #'chunk_memory': "Memory usage for each job (e.g. '100M', '1G') (when saving to binary) - default global"

    sorting_KS4 = run_sorter(sorter_name='kilosort4', recording=rec, folder=save_path, verbose=True, **params, remove_existing_folder=True)                         

    return sorting_KS4


def runKilosort2_5(rec, save_path):
    params = {
        'detect_threshold': 6, 
        'projection_threshold': [10, 4], 
        'preclust_threshold': 8, 
        'whiteningRange': 32.0, 
        'momentum': [20.0, 400.0], 
        'car': True, 
        'minFR': 0.1, 
        'minfr_goodchannels': 0.1, 
        'nblocks': 5, 
        'sig': 20, 
        'freq_min': 150, 
        'sigmaMask': 30, 
        'lam': 10.0, 
        'nPCs': 3, 
        'ntbuff': 64, 
        'nfilt_factor': 4, 
        'NT': None, 
        'AUCsplit': 0.9, 
        'do_correction': True, 
        'wave_length': 61, 
        'keep_good_only': False, 
        'skip_kilosort_preprocessing': False, 
        'scaleproc': None, 
        'save_rez_to_mat': False, 
        'delete_tmp_files': ('matlab_files',), 
        'delete_recording_dat': False, 
        'pool_engine': 'process', 
        'n_jobs': 1, 
        'chunk_duration': '1s', 
        'progress_bar': True, 
        'mp_context': None, 
        'max_threads_per_worker': 1}
    
    Descriptions = {'detect_threshold': 'Threshold for spike detection', 'projection_threshold': 'Threshold on projections', 'preclust_threshold': 'Threshold crossings for pre-clustering (in PCA projection space)', 'whiteningRange': 'Number of channels to use for whitening each channel', 'momentum': 'Number of samples to average over (annealed from first to second value)', 'car': 'Enable or disable common reference', 'minFR': 'Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed', 'minfr_goodchannels': "Minimum firing rate on a 'good' channel", 'nblocks': "blocks for registration. 0 turns it off, 1 does rigid registration. Replaces 'datashift' option.", 'sig': 'spatial smoothness constant for registration', 'freq_min': 'High-pass filter cutoff frequency', 'sigmaMask': 'Spatial constant in um for computing residual variance of spike', 'lam': 'The importance of the amplitude penalty (like in Kilosort1: 0 means not used, 10 is average, 50 is a lot)', 'nPCs': 'Number of PCA dimensions', 'ntbuff': 'Samples of symmetrical buffer for whitening and spike detection', 'nfilt_factor': 'Max number of clusters per good channel (even temporary ones) 4', 'do_correction': 'If True drift registration is applied', 'NT': 'Batch size (if None it is automatically computed)', 'AUCsplit': 'Threshold on the area under the curve (AUC) criterion for performing a split in the final step', 'keep_good_only': "If True only 'good' units are returned", 'wave_length': 'size of the waveform extracted around each detected peak, (Default 61, maximum 81)', 'skip_kilosort_preprocessing': 'Can optionally skip the internal kilosort preprocessing', 'scaleproc': 'int16 scaling of whitened data, if None set to 200.', 'save_rez_to_mat': 'Save the full rez internal struc to mat file', 'delete_tmp_files': "Delete temporary files created during sorting (matlab files and the `temp_wh.dat` file that contains kilosort-preprocessed data). Accepts `False` (deletes no files), `True` (deletes all files) or a Tuple containing the files to delete. Options are: ('temp_wh.dat', 'matlab_files') ", 'delete_recording_dat': "Whether to delete the 'recording.dat' file after a successful run", 'n_jobs': 'Number of jobs (when saving to binary) - default global', 'chunk_size': 'Number of samples per chunk (when saving to binary) - default global', 'chunk_memory': "Memory usage for each job (e.g. '100M', '1G') (when saving to binary) - default global", 'total_memory': "Total memory usage (e.g. '500M', '2G') (when saving to binary) - default global", 'chunk_duration': "Chunk duration in s if float or with units if str (e.g. '1s', '500ms') (when saving to binary) - default global", 'progress_bar': 'If True, progress bar is shown (when saving to binary) - default global'}
    
    sorting_KS2_5 = run_sorter(sorter_name='kilosort2_5', recording=rec, folder=save_path, verbose=True, **params, remove_existing_folder=True)                         

    return sorting_KS2_5



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dat_path = r"E:\Data\mPG_VV\3170_day8_260415_170145\amplifier.dat"
    xml_path = r"E:\Data\mPG_VV\3170_day8_260415_170145\amplifier.xml"
    prm_path = r"E:\Data\mPG_VV\3170_day8_260415_170145\amplifier.prm"
    
    bundled_siteLoc, _, _ = get_prm_info(prm_path)

    probe_l_x = np.zeros(32)    
    probe_l_y = np.arange(32) * 19.5
    linear_siteLoc = np.column_stack((probe_l_x, probe_l_y))[::-1]
    
    print(bundled_siteLoc[:10])
    print(linear_siteLoc[:10])
    
    plt.scatter(bundled_siteLoc[:32, 0], bundled_siteLoc[:32, 1])
    plt.scatter(linear_siteLoc[:, 0], linear_siteLoc[:, 1])

    for i, (x, y) in enumerate(bundled_siteLoc[:32]):
        plt.annotate(str(i), (x, y))
        plt.annotate(str(i), (linear_siteLoc[i, 0], linear_siteLoc[i, 1]))

    plt.show()

    #rec_bundled = create_recording(dat_path, xml_path, bundled_siteLoc, uV_per_count=0.195, shank=2)
    #pre_processed_rec = preProcess_recording(rec_bundled)
    #KS4_sorting = runKilosort4(rec_bundled, save_path=r"E:\Viktor_08_sortings\bundled_KS4_SI2")
    
   #sorting
    