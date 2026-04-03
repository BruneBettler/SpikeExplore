from logging import root

import pandas as pd
import xml.etree.ElementTree as et
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np

def updateXML_skipImped(impedance_excel_path, amplifier_xml_path, threshold=5e06, overwrite=False, output_xml_path=None, verbose=True):
    """ 
    Function marks channels to be skipped in the XML file based on whether the impedance magnitude at 1kHz in Ohms is above the threshold

    Args:
        overwrite (bool, optional): Determines whether or not to overwrite the original XML file. Defaults to False.
        output_xml_path (str, optional): If overwrite == True, contains the file path of the new updated XML file. Defaults to None. If overwrite == True and output_xml_path=None, defaults to updated_amplifier_{datetime.now().strftime(r"%Y%m%d")}.xml"
    
    Returns: path to updated XML file  
    """
    parent_dir = Path(amplifier_xml_path).parent
    
    df = pd.read_excel(impedance_excel_path)

    to_change = [] # store channel indices that will be skipped 
    for row, imped in enumerate(df["Impedance Magnitude at 1000 Hz (ohms)"]):
        # row is the channel number 
        if imped > threshold:
            to_change.append(str(row))
            
    if verbose: print(f"Channels with impedance > {threshold}: {to_change}")
    
    tree = et.parse(amplifier_xml_path)
    root = tree.getroot()

    for channel in root.findall(".//channelGroups/group/channel"):
        if channel.text in to_change:
            channel.set("skip", "1")

    if overwrite: 
        save_path = amplifier_xml_path
    else: 
        if output_xml_path == None:
            save_path = output_xml_path = os.path.join(parent_dir, f"updated_amplifier_{datetime.now().strftime(r"%Y%m%d")}.xml")
    
    
    tree.write(save_path, encoding='utf-8', xml_declaration=True)
    if verbose: print(f"Updated XML file updated and saved to {save_path}")
    
    return save_path




def buildProbeJSON(amplifier_xml_path, xc_location, yc_dist, output_filename, dorsalVentralOrder=False):
    """_summary_

    Args:
        amplifier_xml_path (_type_): _description_
        xc_location (float or array of floats): x-coordinate location in um of probe contact sites for a single shank (float) or multiple (array of floats)
        yc_dist (float or array of floats): distance or depth in um of probe contact sites (y-dir) for a single shank (float) or multiple (array of floats)
        output_filename (_type_): _description_
        dorsalVentralOrientation (bool): indicates whether the 0th entry of the chanMap defines the most dorsal point of the probe. Default False 
    """
    xc = []
    yc = []
    chanMap = []
    n_chan = 0
    shankInd = []
    
    tree = et.parse(amplifier_xml_path)
    root = tree.getroot() 

    for probeIdx, group in enumerate(root.findall(".//channelGroups/group")):
        curr_yDepth = 0.0
        curr_xDepth = xc_location[probeIdx] if not np.isscalar(xc_location) else xc_location
        curr_yc_dist = yc_dist[probeIdx] if not np.isscalar(yc_dist) else yc_dist
        curr_chanMap = []
        curr_yc = []

        for chanIdx, chanData in enumerate(group.findall("channel")):
            if int(chanData.get("skip")) == 0: # we keep this channel
                curr_chanMap.append(int(chanData.text))
                shankInd.append(int(probeIdx)) 
                xc.append(curr_xDepth) # will not need to be flipped 
                curr_yc.append(curr_yDepth)
            
            n_chan += 1 
            curr_yDepth += curr_yc_dist
        
        if dorsalVentralOrder:
            # change the order of the yc, and chanmap for this probe before adding to the larger array
            # shankInd array does not need to be flipped as we are working on a single probe at a time in the loop
            yc.extend(curr_yc[::-1])
            chanMap.extend(curr_chanMap[::-1])
        else: 
            yc.extend(curr_yc)
            chanMap.extend(curr_chanMap)
    
    probe_dict = {
        "xc": xc,
        "yc": yc,
        "kcoords": shankInd,
        "chanMap": chanMap,
        "n_chan": n_chan,
        "shankInd": shankInd,
        "chanMap_orientation_0,1,2,3...": "dorsal_to_ventral" if dorsalVentralOrder else "ventral_to_dorsal"
    } 
    
    kilosort_path = r"C:\Users\social\.kilosort\probes"
    save_path = os.path.join(kilosort_path, output_filename)
    with open(save_path, 'w') as f:
        json.dump(probe_dict, f)
    
    return save_path


def getXMLData(amplifier_xml_path):
    """returns data for SpikeInterface read_binary function from the amplifier XML file.

    Args:
        amplifier_xml_path (_type_): _description_
        shank_idx (int, optional): The shank index for which to extract data. Defaults to 0.
    """
    chanMap = []

    tree = et.parse(amplifier_xml_path)
    root = tree.getroot() 

    fields = ["nBits", "nChannels","samplingRate", "offset", "lfpSamplingRate"] # voltage range and amplification on XML file are not correct (https://intantech.com/files/Intan_RHX_user_guide.pdf)

    out = []

    for tag in fields:
        elem = root.find(f".//{tag}")
        out.append(float(elem.text.strip()) if elem is not None else None)

    for chanData in root.findall(".//channelGroups/group/channel"):
        chanMap.append(int(chanData.text))
      
    return chanMap, [int(out[0]), int(out[1]), float(out[2]), float(out[3]), float(out[4])]


    