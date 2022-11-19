class Alarm:
    def __init__(self):
        self.__limit = 10  # 连续20帧则报警
        self.__climbLimit = 5  # 爬墙的监测降低
        self.__dic = [0, 0, 0]  # 爬墙报警、接触报警

    def climbAlarm(self):
        self.__dic[0] += 1
        self.__dic[1] = 0
        self.__dic[2] = 0
        if self.__dic[0] >= self.__climbLimit:
            print("someone is climbing")

    def touchAlarm(self):
        self.__dic[1] += 1
        self.__dic[0] = 0
        self.__dic[2] = 0
        if self.__dic[1] >= self.__limit:
            print("someones is touching")

    def clear(self):
        self.__dic[2] += 1
        if self.__dic[2] >= self.__limit:
            self.__dic[0] = 0
            self.__dic[1] = 0
