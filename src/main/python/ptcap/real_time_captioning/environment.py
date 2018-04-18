import numpy as np
import torch


class Environment:
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'invalid'
    STATUS_DONE = 'done'
    STATUS_CORRECT_WORD = 'correct'
    STATUS_INCORRECT_WORD = 'incorrect'

    MAX_TOKEN_COUNT = 14
    MAX_FRAME_COUNT = 48

    def __init__(self, encoder, decoder):

        self.encoder = encoder
        self.decoder = decoder

    def reset(self, video, caption):
        self.caption = caption
        self.vid_encoding = self.encoder.extract_features(video)
        self.output_buffer = np.array([0] * self.MAX_TOKEN_COUNT)
        self.input_buffer = []  # input buffer contains the seen video frames
        self.read_count = 0
        self.write_count =0

    def get_state(self):
        #TODO: fix the state
        return {
            "output_buffer": self.output_buffer,
            "input_buffer": self.vid_encoding[0:self.read_count]

        }

    def update_state(self, action):
        status = ""
        if action == 0:  # READ
            if self.read_count == self.MAX_FRAME_COUNT:
                status = Environment.STATUS_INVALID_MOVE
            else:
                self.read_count += 1
                self.input_buffer.append(self.video_encoding[self.read_count])
                status = Environment.STATUS_VALID_MOVE

        if action == 1:  # WRITE
            if self.write_count == self.MAX_CAPTION_COUNT:
                status = Environment.STATUS_INVALID_MOVE
            else:
                word_probs = self.decoder(self.input_buffer, self.caption, use_teacher_forcing=True)
                next_word = torch.max(word_probs)
                self.write_count += 1
                target_word = self.caption[self.write_count]
                if next_word == target_word:
                    return Environment.STATUS_CORRECT_WORD
                if next_word != target_word:
                    return Environment.STATUS_INCORRECT_WORD
                self.output_buffer.append(next_word)

        reward = self.give_reward(status)
        return reward

    def check_finished(self):
        return self.write_count + self.read_count == \
               self.MAX_CAPTION_LEN + self.MAX_FRAME_COUNT

    def give_reward(self, status):
        return {
            Environment.STATUS_VALID_MOVE: 0,
            Environment.STATUS_INVALID_MOVE: -200,
            Environment.STATUS_CORRECT_WORD: 100,
            Environment.STATUS_INCORRECT_WORD: -100
        }[status]