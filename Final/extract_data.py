import re
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import os
import pandas as pd 
import spikeinterface as si
from probeinterface import Probe
import spikeinterface.preprocessing as spre
import xml.etree.ElementTree as et
import ast
from scipy.spatial import cKDTree
import h5py

### JRCLUST 

def get_prm_info(prm_path):
    
    wanted = {"siteLoc", "siteMap", "sampleRate"}
    found = {}

    for line in Path(prm_path).read_text().splitlines():
        line = line.split("%", 1)[0].strip()  # remove MATLAB comments

        if not line or "=" not in line:
            continue

        name, value = map(str.strip, line.split("=", 1))
        name = name.strip()

        if name in wanted:
            found[name] = value.rstrip(";").strip()

        if found.keys() == wanted:
            break

    siteLoc = np.fromstring(found['siteLoc'].replace("[", "").replace("]", "").replace(",", " ").replace(";", " "), sep = " ").reshape(-1, 2)

    siteMap = np.fromstring(found["siteMap"].strip("[]"), sep=",").astype("int32") - 1

    sampleRate = float(found["sampleRate"])

    return siteLoc, siteMap, sampleRate

def identical_spikes(spike_times_A, spike_times_B, tol=0):
        a = np.asarray(spike_times_A)
        b = np.asarray(spike_times_B)

        ia = np.argsort(a)
        ib = np.argsort(b)

        a_sorted = a[ia]
        b_sorted = b[ib]

        i = j = 0
        matches = []

        while i < a_sorted.size and j < b_sorted.size:
            d = a_sorted[i] - b_sorted[j]

            if abs(d) <= tol:
                matches.append((ia[i], ib[j]))  # original indices
                i += 1
                j += 1
            elif d < -tol:
                i += 1
            else:
                j += 1

        return np.array(matches, dtype=int)
    
    
def extract_cluster_data_JRCLUST(res_mat_path, siteMap):
    res_file = Path(res_mat_path)

    with h5py.File(res_file, "r") as f:
        spike_clusters = f["spikeClusters"][...]
        print(f"Total spike num: {spike_clusters.shape}")
        cluster_sites = f["clusterSites"][...]
    
    
    cluster_sites = cluster_sites.squeeze().ravel().astype("int32") - 1 # now in python index in sitemap where the channel is located
    spike_clusters = np.asarray(spike_clusters).squeeze().ravel().astype(np.int32)
    unique_cluster_ids = np.unique(spike_clusters[spike_clusters > 0])

    return np.stack((unique_cluster_ids, siteMap[cluster_sites], cluster_sites))

    
def extract_unit_data_JRCLUST(res_mat_path, spikesFilt_mat_path, siteMap, cluster_id):
    """
    return dict{
        unit_peak_chan: in siteMap channel values 
        Spike times:
        Spike amplitudes:
        Spike location: 
        Spike Waveform: 
    }
    """
    return_dict = {}
    
    res_file = Path(res_mat_path)

    with h5py.File(res_file, "r") as f:
        spike_clusters = f["spikeClusters"][...].squeeze().ravel().astype("int32") # in MATLAB INDEXING 
        cluster_sites = f["clusterSites"][...].squeeze().ravel().astype("int32") - 1 # now in python index in sitemap where the channel is located
        spike_times = f["spikeTimes"][...].squeeze() - 1 # TODO VERIFY THIS 
    
        cluster_ids = np.unique(spike_clusters)
        cluster_ids = cluster_ids[cluster_ids > 0] # keep only valid JRCLUST ids 
        
        mask = spike_clusters == cluster_id
    
        return_dict["unit_peak_chan_index"] = cluster_sites[np.where(cluster_ids == cluster_id)[0]]
        return_dict["spike_times"] = spike_times[mask]
        
        return_dict["spike_amplitudes"] = f["spikeAmps"][...].squeeze()[mask]
        return_dict["spike_locations"] = f["spikePositions"][...].T[mask]
    
    if h5py.is_hdf5(spikesFilt_mat_path):
        data = h5py.File(spikesFilt_mat_path, "r")
    else:
        data = loadmat(spikesFilt_mat_path)
        res = data['res']['spikesFilt'][0,0]
        return_dict["spike_waveforms"] = res[:, :, mask].T #/ 0.195 # convert to uV using gain
        
    return return_dict

### KILOSORT

def getXMLData(amplifier_xml_path):
    """returns data for SpikeInterface read_binary function from the amplifier XML file.

    Args:
        amplifier_xml_path (_type_): _description_
        shank_idx (int, optional): The shank index for which to extract data. Defaults to 0.
    """
    chanMap = []
    skippedChannels = []

    tree = et.parse(amplifier_xml_path)
    root = tree.getroot() 

    fields = ["nBits", "nChannels","samplingRate", "offset", "lfpSamplingRate"] # voltage range and amplification on XML file are not correct (https://intantech.com/files/Intan_RHX_user_guide.pdf)

    out = []

    for tag in fields:
        elem = root.find(f".//{tag}")
        out.append(float(elem.text.strip()) if elem is not None else None)

    for chanData in root.findall(".//channelGroups/group/channel"):
        chanID = int(chanData.text)
        chanMap.append(chanID)
        if int(chanData.get("skip")) == 1: # add channel to skipped array
                skippedChannels.append(chanID)

    return chanMap, skippedChannels, [int(out[0]), int(out[1]), float(out[2]), float(out[3]), float(out[4])]

def get_unit_ids_KILO_res(results_dir):
    # if chans.txt exists, only return these 
    path = Path(os.path.join(results_dir, "chans.txt"))

    chans = ast.literal_eval(path.read_text().strip()) if path.exists() else np.unique(np.load(os.path.join(results_dir, 'spike_clusters.npy')))
    return chans    
    
def get_unit_ids_KILO_ana(analyzer):
    return analyzer.sorting.get_unit_ids()


def extract_data_kilosort_analyzer(analyzer, unit_index, ms_before=0.5, ms_after=1.5):
    """
    return dict{
        unit_peak_chan: in siteMap channel values 
        Spike times:
        Spike amplitudes:
        Spike location: 
        Spike Waveform: 
    }
    """
    return_dict = {}
    
    recording = analyzer.recording
    sorting = analyzer.sorting
    
    unit_ids = analyzer.sorting.get_unit_ids()
    unit_id = unit_ids[unit_index]

    fs = recording.get_sampling_frequency()
    n_before = int(ms_before / 1000 * fs)
    n_after = int(ms_after / 1000 * fs)
    n_samples = n_before + n_after

    spike_times = sorting.get_unit_spike_train(unit_id).astype(np.int64)

    valid = (spike_times >= n_before) & (spike_times + n_after <= recording.get_num_frames())
    spike_times = spike_times[valid]
    

    if len(spike_times) > 500:
        spike_times_peak = np.random.choice(spike_times, 500, replace=False)
        spike_times_peak.sort()
    else:
        spike_times_peak = spike_times

    # extract all channels for a subset, only to find best channel
    wfs = []
    for t in spike_times_peak:
        wfs.append(
            recording.get_traces(
                start_frame=int(t - n_before),
                end_frame=int(t + n_after),
                return_in_uV=True,
            )
        )

    wfs = np.asarray(wfs)  # spikes x samples x channels

    mean_wf = wfs.mean(axis=0)
    peak_channel_ind = np.ptp(mean_wf, axis=0).argmax()
    peak_channel_id = recording.get_channel_ids()[peak_channel_ind]

    waveforms = np.empty((len(spike_times), n_samples), dtype=np.float32)

    for i, t in enumerate(spike_times):
        waveforms[i] = recording.get_traces(
            start_frame=int(t - n_before),
            end_frame=int(t + n_after),
            channel_ids=[peak_channel_id],
            return_in_uV=True
        )[:, 0]
    
    locs = analyzer.get_extension("spike_locations").get_data()
    spks = analyzer.sorting.to_spike_vector()
    spike_locations = locs[spks["unit_index"] == unit_index] 
    
    amps_by_unit = analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit", concatenated=True)


    return_dict["spike_times"] = spike_times
    return_dict["spike_waveforms"] = waveforms
    return_dict["unit_peak_chan_index"] = peak_channel_ind # siteMap[cluster_sites][unit_index]
    return_dict["spike_locations"] = np.column_stack([spike_locations["x"], spike_locations["y"]])
    return_dict["spike_amplitudes"] = amps_by_unit[unit_id]
    
    return return_dict
    
    
    
def extract_data_kilosort_results(recording, result_dir_path, unit_id=None, unit_index=None, ms_before=0.5, ms_after=1.5):
    """
    unit_id = directly the id found in np.unique(clu)
    unit_index = index 
    return dict{
        unit_peak_chan: in siteMap channel values 
        Spike times:
        Spike amplitudes:
        Spike location: 
        Spike Waveform: 
    }
    """
    return_dict = {}
    
    results_dir = Path(result_dir_path)
    
    chan_map =  np.load(os.path.join(results_dir, 'channel_map.npy'))

    clu = np.load(os.path.join(results_dir, 'spike_clusters.npy'))
    unit_ids = np.unique(clu)
    
    if unit_index != None:
        unit_id = unit_ids[unit_index]

    templates =  np.load(os.path.join(results_dir, 'templates.npy'))

    chan_best_a = ( templates**2).sum(axis=1).argmax(axis=-1)
    chan_best_b = np.ptp(templates, axis=1).argmax(axis=1)

    amplitudes = np.load(os.path.join(results_dir, 'amplitudes.npy'))
    st = np.load(os.path.join(results_dir, 'spike_times.npy'))

    spike_positions = np.load(os.path.join(results_dir, 'spike_positions.npy'))

    fs = recording.get_sampling_frequency()

    n_before = int(ms_before / 1000 * fs)
    n_after = int(ms_after / 1000 * fs)

    indices = np.where(clu == unit_id)[0]
    
    # peak channel 
    peak_channel_id = chan_best_b[unit_id] # chan_best a or b 
    peak_channel = chan_map[peak_channel_id]
  
    # for each spike, extract the waveform on the peak channel 
    waveforms = np.empty((len(indices), n_before + n_after), dtype=np.float32)
    # they will be ordered by time since that's how the st and clu arrays are ordered

    for i, spike_idx in enumerate(indices):                 
        spike_time = st[spike_idx]

        start = spike_time - n_before
        end = spike_time + n_after

        waveforms[i] = recording.get_traces(start_frame=start, end_frame=end, channel_ids=[peak_channel], return_in_uV=True)[:, 0]

    waveforms = np.asarray(waveforms)

    return_dict["unit_peak_chan_index"] = peak_channel_id # in index of chanMap
    return_dict["spike_times"] = st[indices]
    return_dict["spike_amplitudes"] = amplitudes[indices]
    return_dict["spike_locations"] = spike_positions[indices]
    return_dict["spike_waveforms"] = waveforms
    
    return return_dict

def _mode_int(x):
    x = np.asarray(x, dtype=int)
    return np.bincount(x).argmax()

def get_kilo_rec_unit_peak_channels(results_dir, filename="chans.txt"):
    results_dir = Path(results_dir)

    clu = np.load(results_dir / "spike_clusters.npy")
    templates = np.load(results_dir / "templates.npy")
    chan_map = np.load(results_dir / "channel_map.npy")

    spike_templates_path = results_dir / "spike_templates.npy"
    if spike_templates_path.is_file():
        spike_templates = np.load(spike_templates_path)
    else:
        spike_templates = clu

    path = Path(os.path.join(results_dir, filename))

    chans = ast.literal_eval(path.read_text().strip()) if path.exists() else np.unique(np.load(os.path.join(results_dir, 'spike_clusters.npy')))

    template_peak_chan_idx = np.ptp(templates, axis=1).argmax(axis=1)
    template_peak_chan = chan_map[template_peak_chan_idx]

    out = []

    for unit_id in chans:
        spike_idx = np.flatnonzero(clu == unit_id)
        if len(spike_idx) == 0:
            continue

        template_id = _mode_int(spike_templates[spike_idx])
        out.append(np.array([unit_id, int(template_peak_chan[template_id]), int(template_peak_chan_idx[template_id])]))  # unit id, peak channel, peak channel index

    return np.array(out)

import numpy as np


def cluster_units_by_channel_radius(unit_arrays, chan_map, radius=0):
    """
    Parameters
    ----------
    unit_arrays : list[np.ndarray]
        One array per recording, each shape (n_units, 3):
        [unit_id, peak_chan_id, peak_chan_index]

    chan_map : array-like
        Ordered channel IDs. Radius is interpreted in this channel-map order.

    radius : int
        Number of channels away on either side.
        radius=0 means same channel only.
        radius=1 means ±1 channel index.

    Returns
    -------
    clusters : list[list[tuple]]
        Each cluster contains tuples:
        (recording_index, unit_id, peak_chan_id, peak_chan_index)
    """

    chan_map = np.asarray(chan_map)
    chan_id_to_order = {int(ch): i for i, ch in enumerate(chan_map)}

    all_units = []

    for rec_i, arr in enumerate(unit_arrays):
        arr = np.asarray(arr)

        for unit_id, peak_chan_id, peak_chan_index in arr:
            chan_order = chan_id_to_order[int(peak_chan_id)]

            all_units.append((
                chan_order,
                rec_i,
                int(unit_id),
                int(peak_chan_id),
                int(peak_chan_index),
            ))

    all_units.sort(key=lambda x: x[0])

    clusters = []
    used = np.zeros(len(all_units), dtype=bool)

    for i, u in enumerate(all_units):
        if used[i]:
            continue

        center_chan_order = u[0]
        cluster = []
        used[i] = True

        for j in range(i, len(all_units)):
            v = all_units[j]

            if v[0] - center_chan_order > radius:
                break

            if abs(v[0] - center_chan_order) <= radius:
                cluster.append(v[1:])
                used[j] = True

        clusters.append(cluster)

    return clusters # shape (recording_index, unit_id, peak_chan_id, peak_chan_index)



    

def get_recording(dat_path, xml_path, siteLoc):
    channel_ids, skippedChannels, xml_data = getXMLData(xml_path)
    nBits, nChannels, samplingRate, offset, lfpSamplingRate = xml_data
    num_contact_sites = 32
    #samplingRate = 20000 
    uV_per_count = 0.195 #https://intantech.com/files/Intan_RHX_user_guide.pdf
    full_recording = si.read_binary(file_paths=dat_path, sampling_frequency=samplingRate, dtype=f"int{nBits}", num_channels=nChannels, channel_ids=np.arange(nChannels), is_filtered=False, gain_to_uV=uV_per_count, offset_to_uV=0.0)
    shank2_channel_ids = np.array(channel_ids[32:])
    shank2_recording = full_recording.select_channels(channel_ids=shank2_channel_ids)

    probe_m_x = siteLoc[:32,0] 
    probe_m_y = siteLoc[:32,1] 

    probe_m = Probe(ndim=2, si_units='um')
    probe_m.set_contacts(positions=np.column_stack((probe_m_x, probe_m_y)), shapes=np.array(['square'] * num_contact_sites), shape_params={"width": 13})
    # set the contact IDs for each site 
    probe_m.set_device_channel_indices(np.arange(num_contact_sites))
    probe_m.set_contact_ids(shank2_channel_ids)
    probe_m.create_auto_shape()
    # link the probe geometry to our recording object 
    shank2_probe_m = shank2_recording.set_probe(probe_m, in_place=False)
    shank2_skipped_channels = np.intersect1d(shank2_channel_ids, skippedChannels)

    m_rec_clean = shank2_probe_m.remove_channels(shank2_skipped_channels)

    # common preprocessing before two shanks are applied
    P1 = spre.center(m_rec_clean, mode='median', dtype='float32')
    P2 = spre.highpass_filter(P1, freq_min=300.0) # TODO: consider bandpass 
    P3_loc = spre.common_reference(recording = P2, reference='local', operator='median', local_radius=(40, 180), min_local_neighbors=5)
    
    return P3_loc



if __name__ == "__main__":
    a = get_unit_ids_spikesFilt(r"E:\Viktor_08_sortings\bundled_JRCLUST\bundled_spikesFilt.mat")
    
    
    # JRCLUST 
    #prm_path = r"E:\Data\mPG_VV\3170_day8_260415_170145\amplifier.prm"
    #JRC_res_path = r"E:\Data\mPG_VV\3170_day8_260415_170145\res_waveforms.mat"
    
    #siteLoc_JR, siteMap_JR, sampleRate = get_prm_info(prm_path)
    #unit_ids = get_unit_ids(JRC_res_path)
    #print(unit_ids)
    #unit_dict_JR = extract_data_JRCLUST(siteMap_JR, JRC_res_path, unit_ids[0])
    
    # Kilosort
   # KS4_sorted_path_B = r"E:\Viktor_08_sortings\bundled_KS4_KS" #"C:\Users\social\Desktop\temp_Brune\viktor08_kilosort4"
    #KS4_sorted_path_L = r"E:\Viktor_08_sortings\linear_KS4_KS"
    
    #dat_path = r"E:\Data\mPG_VV\3170_day8_260415_170145\amplifier.dat"
    #xml_path = r"E:\Data\mPG_VV\3170_day8_260415_170145\amplifier.xml"
    
    #unit_ids = get_unit_ids_KILO_res(KS4_sorted_path)
    #print(unit_ids)

    #recording = get_recording(dat_path, xml_path, siteLoc=siteLoc_JR)
    
    #unit_dict_KILO = extract_data_kilosort_results(recording, KS4_sorted_path, unit_id=2)
    #print("done")
    
    #out_B = get_kilo_rec_unit_peak_channels(KS4_sorted_path_B, filename="chans.txt")
    #out_L = get_kilo_rec_unit_peak_channels(KS4_sorted_path_L, filename="chans.txt")
    
    #clusters = cluster_units_by_channel_radius(unit_arrays=[out_B, out_L], chan_map=siteMap_JR-1, radius=0)
   # print(len(clusters))
    
    #unit_index = 4
    #KILO_analyzer_path = r"C:\Users\social\Desktop\temp_Brune\SpikeExplore\analyzers\KS4_M_curated"
    #analyzer = si.load_sorting_analyzer(KILO_analyzer_path)
    #extract_data_kilosort_analyzer(analyzer, unit_index)

    
    