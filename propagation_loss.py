import math

class Propagation_Loss:
    def __init__(self, access_point_set, terminal_set):
        self.access_point_set = access_point_set
        self.terminal_set = terminal_set

    def free_space_path_loss(self):
        tssi = 20
        noise = -100
        bandwidth = 100
        distance = 1

        fspl_inspection = self.fspl_calculation(distance)
        rssi_inspection = self.fspl_rssi_calculation(fspl_inspection, tssi)
        snr_inspection = self.snr_calculation(rssi_inspection, noise)
        shannon_inspection = self.shannon_calculation(snr_inspection, bandwidth)

        rssi_all = []
        access_point_terminal_relation_all = []
        for terminal in self.terminal_set:
            rssi_set = []
            access_point_terminal_relation_set = []
            for access_point in self.access_point_set:
                distance = self.distance(access_point.position, terminal.position)
                fspl = self.fspl_calculation(distance)
                rssi = self.fspl_rssi_calculation(fspl, tssi)
                snr = self.snr_calculation(rssi, noise)
                shannon = self.shannon_calculation(snr, bandwidth)
                shannon_rate = shannon / shannon_inspection

                rssi_set.append(rssi)
                access_point_terminal_relation = self.FSPL(shannon_rate)
                access_point_terminal_relation_set.append(access_point_terminal_relation)
            rssi_all.append(rssi_set)
            access_point_terminal_relation_all.append(access_point_terminal_relation_set)
        
        return rssi_all, access_point_terminal_relation_all

    def non_free_space_path_loss(self):
        tssi = 20
        noise = -100
        bandwidth = 100
        distance = 1

        fspl_inspection = self.fspl_calculation(distance)
        rssi_inspection = self.fspl_rssi_calculation(fspl_inspection, tssi)
        snr_inspection = self.snr_calculation(rssi_inspection, noise)
        shannon_inspection = self.shannon_calculation(snr_inspection, bandwidth)

        rssi_all = []
        access_point_terminal_relation_all = []
        for terminal in self.terminal_set:
            rssi_set = []
            access_point_terminal_relation_set = []
            for access_point in self.access_point_set:
                distance = self.distance(access_point.position, terminal.position)
                fspl = self.fspl_calculation(distance)
                wall_loss = self.wall_loss_calculation(terminal.position)
                rssi = self.nfspl_rssi_calculation(fspl, wall_loss, tssi)
                snr = self.snr_calculation(rssi, noise)
                shannon = self.shannon_calculation(snr, bandwidth)
                shannon_rate = shannon / shannon_inspection

                rssi_set.append(rssi)
                access_point_terminal_relation = self.FSPL(shannon_rate)
                access_point_terminal_relation_set.append(access_point_terminal_relation)
            rssi_all.append(rssi_set)
            access_point_terminal_relation_all.append(access_point_terminal_relation_set)
        
        return rssi_all, access_point_terminal_relation_all

    def distance(self, access_point_position, terminal_position):
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(access_point_position, terminal_position)))

    def fspl_calculation(self, distance):
        return 20 * math.log10(4 * math.pi * distance * 5.0e9 / 299792458)
    
    def wall_loss_calculation(self, terminal_position):
        wall_loss = 20
        if terminal_position[1] <= -17 or 17 <= terminal_position[1]:
            return wall_loss
        else:
            return 0

    def fspl_rssi_calculation(self, fspl, tssi):
        return tssi - fspl
    
    def nfspl_rssi_calculation(self, fspl, wall_loss, tssi):
        return tssi - fspl - wall_loss

    def snr_calculation(self, rssi, noise):
        snr = rssi - noise
        return math.pow(10, snr / 10)

    def shannon_calculation(self, snr,band):
        return band * math.log2(1 + snr)
    
    class FSPL:
        def __init__(self, rate):
            self.rate = rate
    
    class NFSPL:
        def __init__(self, rate):
            self.rate = rate