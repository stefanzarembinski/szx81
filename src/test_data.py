from os import path
from core import *

class TestData:
    """
Reads data from all files in a `test_data`, specified with the combined definitions in
the `SETUP` and CONFIG maps.
Loads the data to a list accesible with a generator defined in the Class.
    """
    def __init__(self, data_count=200):
        data_count += WARMUP_TIME
        timestring_format = config.CONFIG['timestring']
        data = {} # `using dict` to avoid duplicates
        self.count = 0
        data_dir = path.join(TEST_DATA_DIR, TEST_DATA)
        for data_file in os.listdir(data_dir):
            with open(path.join(data_dir, data_file), newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                next(spamreader)
                for line in spamreader:
                    if data_count is not None:
                        if data_count <= 0:
                            break
                        data_count -= 1

                    line = line[0] + ' ' + line[1]
                    values = line.split(',')
                    timestring = values[0]
                    strptime = datetime.datetime.strptime(timestring, timestring_format)
                    timestamp = time.mktime(strptime.timetuple())
                    if not timestamp % config.CONFIG['period'] == 0:
                        raise RuntimeError('Wrong data period!')

                    _values = []
                    for i in range(1, len(values) - 1):
                        _values.append(float(values[i]))
                    data[timestamp] = _values

        # Sort data because datafile order can be not sequential:
        data = sorted(data.items())

        # Subtract mooving avarage:
        start_time = None
        ma = MovingAverage()
        self.data = []
        for dat in data:
            timestamp = dat[0]
            if start_time is None:
                start_time = timestamp
            ma.add(dat[1][0])

            if (timestamp - start_time) / 60 < WARMUP_TIME:
                continue
            # import pdb; pdb.set_trace()
            mean_value = ma.ma()
            values = [val - mean_value for val in dat[1]]
            self.data.append((timestamp, values))
    
    def reset_generator(self, count=0):
        if count != 0:
            if isinstance(count, numbers.Number):
                timestamp = int(count)
                for index in range(len(self.data) - 1, -1, -1):
                    if self.data[index][0] <= timestamp: break
                count = index
            
        self.count = count

    def generator(self):
        if self.count < len(self.data):
            retval = self.data[self.count]
            self.count += 1 
            yield retval
        yield None
        
    def plot(self, count=None):
        plot_values = self.data[:count] if count is not None else self.data
        x = []
        y = []
        _x = 0
        deltax = config.CONFIG['period'] / 60
        for _xy in plot_values:
            x.append(_x)
            _x += deltax
            y.append(_xy[1][1])

        plt.plot(x, y, label='forex', color='green')
        plt.xlabel('time [min]')
        plt.ylabel('open value')
        plt.legend()
        plt.show()

TEST_DATA_OBJ = None
DATA = None
TIME_COUNT = None
VALUE = None
TIMESTAMP = None

def init(data_size=3000):
    global TEST_DATA_OBJ
    global DATA
    global TIME_COUNT
    global VALUE
    global TIMESTAMP

    TEST_DATA_OBJ = TestData(data_size)
    DATA = TEST_DATA_OBJ.data # 86401 builtin_function_or_method' object is not subscriptable
    TIME_COUNT = np.array([i for i in range(len(DATA))], dtype='float64')
    VALUE = np.array([values[1][0] for values in DATA])
    TIMESTAMP = np.array([values[0] for values in DATA])
    print(f'Test data size is {len(DATA)}')

init()

def main():
    TEST_DATA_OBJ.plot(12000)

if __name__ == "__main__":
    main()
