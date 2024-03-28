#! /bin/env python
#
# Michael Gibson 17 July 2015

import os, time
import numpy as np

from intanutil.read_header import read_header
from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
from intanutil.read_one_data_block import read_one_data_block
from intanutil.notch_filter import notch_filter
from intanutil.data_to_result import data_to_result
import multiprocessing as mp

class RHDError(Exception):
    pass

def notch_filter_mp(arr, index,  sr, nff):
    arr = notch_filter(arr, sr, nff, 10)
    print('Electrode ', index, 'filtered')
    
def read_data(filename, parallelFlg=True):
    """Reads Intan Technologies RHD2000 data file generated by evaluation board GUI.
    
    Data are returned in a dictionary, for future extensibility.
    """

    tic = time.time()
    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)

    header = read_header(fid)

    print('Found {} amplifier channel{}.'.format(header['num_amplifier_channels'],
                                                 plural(header['num_amplifier_channels'])))

    print('Found {} auxiliary input channel{}.'.format(
        header['num_aux_input_channels'],
        plural(header['num_aux_input_channels'])))

    print('Found {} board ADC channel{}.'.format(header['num_board_adc_channels'],
                                                 plural(header['num_board_adc_channels'])))

    print('Found {} board digital input channel{}.'.format(
        header['num_board_dig_in_channels'],
        plural(header['num_board_dig_in_channels'])))

    print('Found {} supply voltage channel{}.'.format(
        header['num_supply_voltage_channels'],
        plural(header['num_supply_voltage_channels'])))
    if header["num_supply_voltage_channels"] > 0:
        print("Supply voltage data found but will not be imported.")

    print('Found {} board digital output channel{}.'.format(
        header['num_board_dig_out_channels'],
        plural(header['num_board_dig_out_channels'])))
    if header["num_board_dig_out_channels"] > 0:
        print("Digital output data found but will not be imported.")

    print('Found {} temperature sensors channel{}.'.format(
        header['num_temp_sensor_channels'],
        plural(header['num_temp_sensor_channels'])))
    if header["num_temp_sensor_channels"] > 0:
        print("Temperature sensor data found but will not be imported.")

    print('')

    # Determine how many samples the data file contains.
    bytes_per_block = get_bytes_per_data_block(header)

    # How many data blocks remain in this file?
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    if bytes_remaining % bytes_per_block != 0:
        raise RHDError('Something is wrong with file size : should have a whole number of data blocks')

    num_data_blocks = int(bytes_remaining / bytes_per_block)

    num_amplifier_samples = 60 * num_data_blocks
    num_aux_input_samples = 15 * num_data_blocks
    num_board_adc_samples = 60 * num_data_blocks
    num_board_dig_in_samples = 60 * num_data_blocks
    # num_supply_voltage_samples = 1 * num_data_blocks
    # num_board_dig_out_samples = 60 * num_data_blocks

    record_time = num_amplifier_samples / header['sample_rate']

    if data_present:
        print('File contains {:0.3f} seconds of data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(record_time, header['sample_rate'] / 1000))
    else:
        print('Header file contains no data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(header['sample_rate'] / 1000))

    if data_present:
        # Pre-allocate memory for data.
        print('')
        print('Allocating memory for data...')

        data = {}
        if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=np.int)
        else:
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=np.uint)

        data['amplifier_data'] = np.zeros([header['num_amplifier_channels'],
                                           num_amplifier_samples],
                                          dtype=np.float32)
        data['aux_input_data'] = np.zeros([header['num_aux_input_channels'],
                                           num_aux_input_samples],
                                          dtype=np.float32)
        # data['supply_voltage_data'] = np.zeros([header['num_supply_voltage_channels'],
        #                                         num_supply_voltage_samples],
        #                                        dtype=np.float32)
        # data['temp_sensor_data'] = np.zeros([header['num_temp_sensor_channels'],
        #                                      num_supply_voltage_samples],
        #                                     dtype=np.float32)
        data['board_adc_data'] = np.zeros([header['num_board_adc_channels'],
                                           num_board_adc_samples],
                                          dtype=np.float32)
        data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'],
                                              num_board_dig_in_samples],
                                             dtype=np.uint)
        data['board_dig_in_raw'] = np.zeros(num_board_dig_in_samples, dtype=np.uint)
        # data['board_dig_out_data'] = np.zeros([header['num_board_dig_out_channels'],
        #                                        num_board_dig_out_samples],
        #                                       dtype=np.uint)
        # data['board_dig_out_raw'] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

        # Read sampled data from file.
        print('Reading data from file...')

        # Initialize indices used in looping
        indices = {}
        indices['amplifier'] = 0
        indices['aux_input'] = 0
        indices['supply_voltage'] = 0
        indices['board_adc'] = 0
        indices['board_dig_in'] = 0
        indices['board_dig_out'] = 0

        print_increment = 10
        percent_done = print_increment
        for i in range(num_data_blocks):
            read_one_data_block(data, header, indices, fid)

            # Increment indices
            indices['amplifier'] += 60
            indices['aux_input'] += 15
            indices['supply_voltage'] += 1
            indices['board_adc'] += 60
            indices['board_dig_in'] += 60
            indices['board_dig_out'] += 60      

            fraction_done = 100 * (1.0 * i / num_data_blocks)
            if fraction_done >= percent_done:
                print('{}% done...'.format(percent_done))
                percent_done = percent_done + print_increment

        # Make sure we have read exactly the right amount of data.
        bytes_remaining = filesize - fid.tell()
        if bytes_remaining != 0:
            raise RHDError('Error: End of file not reached.')



    # Close data file.
    fid.close()

    if (data_present):
        print('Parsing data...')

        # Extract digital input channels to separate variables.
        for i in range(header['num_board_dig_in_channels']):
            data['board_dig_in_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_in_raw'], (1 << header['board_dig_in_channels'][i]['native_order'])), 0)
        #
        # # Extract digital output channels to separate variables.
        # for i in range(header['num_board_dig_out_channels']):
        #     data['board_dig_out_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_out_raw'], (1 << header['board_dig_out_channels'][i]['native_order'])), 0)

        # Check for gaps in timestamps.
        num_gaps = np.sum(np.not_equal(data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
        if num_gaps == 0:
            print('No missing timestamps in data.')
        else:
            print('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(num_gaps))

        # Scale time steps (units = seconds).
        data['t_amplifier'] = data['t_amplifier']/header['sample_rate']    # This is not done in place because it needs to be recasted as a float
        data['t_aux_input'] = data['t_amplifier'][range(0, len(data['t_amplifier']), 4)]
        # data['t_supply_voltage'] = data['t_amplifier'][range(0, len(data['t_amplifier']), 60)]
        
        # We don't need 4 copies of the array
        
        #data['t_board_adc'] = data['t_amplifier']
        #data['t_dig'] = data['t_amplifier']
        #data['t_temp_sensor'] = data['t_supply_voltage']

        # If the software notch filter was selected during the recording, apply the
        # same notch filter to amplifier data here.
        # This is slow trying multiprocessing.
        if header['notch_filter_frequency'] > 0:
            print('Applying notch filter...')
                       
            if parallelFlg:
                # Set up the data and the arguments for parallel processing
                arglist = []
                arrlist = []
                for i in range(header['num_amplifier_channels']):
                    arrlist.append(mp.Array('f', data['amplifier_data'][i], lock=False))
                    arglist.append((arrlist[i], i, header['sample_rate'], header['notch_filter_frequency']))
            
                # Spawn the process
                proclist = []
                for i in range(header['num_amplifier_channels']):
                    p = mp.Process(target=notch_filter_mp, args=arglist[i])
                    p.start()
                    proclist.append(p) 
            
                # Wait for the process to finish and copy the data
                for i in range(header['num_amplifier_channels']):
                    proclist[i].join()
                    data['amplifier_data'][i] = arrlist[i]            
            
            else:
                print_increment = 10
                percent_done = print_increment
                for i in range(header['num_amplifier_channels']):
                    data['amplifier_data'][i,:] = notch_filter(data['amplifier_data'][i,:], header['sample_rate'], header['notch_filter_frequency'], 10)

                    fraction_done = 100 * (i / header['num_amplifier_channels'])
                    if fraction_done >= percent_done:
                        print('{}% done...'.format(percent_done))
                        percent_done += print_increment
    else:
        data = []

    # Move variables to result struct.
    result = data_to_result(header, data, data_present)

    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))
    return result

def plural(n):
    """Utility function to optionally pluralize words based on the value of n.
    """

    if n == 1:
        return ''
    else:
        return 's'

#if __name__ == '__main__':
#    a=read_data(sys.argv[1])
#    #print a
    
# fname = '/Users/frederictheunissen/Documents/Data/Chronic V2/HPG8003/hpg8003_day_2_160503_095828.rhd'
# result = read_data(fname)


