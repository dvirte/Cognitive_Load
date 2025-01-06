
class MazeLevels:
    def __init__(self, ExpData, DataObj):
        self.num_levels = len(ExpData.play_periods)
        self.rest_periods_csv, self.play_periods_csv, self.calibrate_periods_csv = (
            self.find_periods(DataObj.read_triggers_csv()))

    def find_periods(self, triggers_dict):
        """
        Identifies the rest periods based on triggers and their corresponding time stamps.
        :param triggers_dict: dictionary containing the trigger_id and timestamp keys.
        :return: rest_periods, play_periods, calibrate_periods
        """
        rest_periods = []
        play_periods = []
        calibrate_periods = []
        end_idx = None
        cal_ind = None

        for i, trigger in enumerate(triggers_dict['trigger_id']):
            if trigger == 6 or trigger == 9:
                start_idx = i
                if end_idx is not None:
                    play_periods.append((triggers_dict['timestamp'][end_idx],
                                         triggers_dict['timestamp'][start_idx]))
            elif trigger == 4 or trigger == 5:
                end_idx = i
                rest_periods.append((triggers_dict['timestamp'][start_idx],
                                     triggers_dict['timestamp'][end_idx]))
                if trigger == 5:
                    break

            elif 13 <= trigger <= 18:
                if cal_ind is not None:
                    calibrate_periods.append((triggers_dict['timestamp'][cal_ind],
                                              triggers_dict['timestamp'][i]))
                cal_ind = i

        if trigger != 5:
            rest_periods.append((triggers_dict['timestamp'][end_idx],
                                 triggers_dict['timestamp'][-1]))
        return rest_periods, play_periods, calibrate_periods
