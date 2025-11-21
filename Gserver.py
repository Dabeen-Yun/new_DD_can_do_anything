class Gserver:
    def __init__(self, id, lon, lat, vsg_id):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.vsg_id = vsg_id
        self.vnf_list = []  # 클래스 속성 초기화

        # transmit queue
        self.queue_TSL = []
        # processing queue
        self.process_queue = []