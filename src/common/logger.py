
MAX_HISTORY_SIZE = 10
FLUSH_DELTA = 1000


class Logger:

    def __init__(self, agent, out_filename: str):
        self.agent = agent
        self.container = agent.container
        self.network = agent.network
        self.movements = []
        self.log_content = []
        self.out_filename = out_filename
        self.last_log_size = 0

    def write(self):
        current_tick = self.network.current_tick
        self.movements.append((current_tick, self.agent.actions))
        self.log_content.append(f'{current_tick}: actions {self.agent.actions}')
        if len(self.movements) > MAX_HISTORY_SIZE:
            del self.movements[0]

    def write_content(self, info):
        current_tick = self.agent.network.current_tick
        self.log_content.append(f'{current_tick}: {info}')
        self.flush()

    def flush(self):
        if len(self.log_content) > self.last_log_size + FLUSH_DELTA:
            self.save_list_to_file(self.log_content, self.out_filename)
            self.last_log_size = len(self.log_content)

    @staticmethod
    def save_list_to_file(lines, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for line in lines:
                print(str(line).strip(), file=file)
