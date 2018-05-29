from src.model.model import Model
import tensorflow as tf
import easy_tf_log


class TensorflowBasedModel(Model):
    def __init__(self, config, data=None):
        super().__init__(config, data)
        self.var_list = []
        self._saver = None
        self._memory_length = 0

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=0, var_list=self.var_list)
        return self._saver

    @saver.setter
    def saver(self, new_val):
        self._saver = new_val

    @property
    def memory_length(self):
        return self._memory_length

    def save_snapshot(self, *args, **kwargs):
        sess = tf.get_default_session()
        sess.run(self.save_snapshot_op)
        super().save_snapshot(*args, **kwargs)

    def load_snapshot(self, *args, **kwargs):
        sess = tf.get_default_session()
        sess.run(self.load_snapshot_op)
        super().load_snapshot(*args, **kwargs)

    def init(self):
        sess = tf.get_default_session()
        with tf.variable_scope('snapshot'):
            for var in self.var_list:
                snap_var = tf.Variable(initial_value=sess.run(var), expected_shape=var.get_shape().as_list())
                self.snapshot_var.append(snap_var)
                self.save_snapshot_op.append(tf.assign(var, snap_var))
                self.load_snapshot_op.append(tf.assign(snap_var, var))

        sess.run(tf.variables_initializer(var_list=self.snapshot_var))
        sess.run(self.save_snapshot_op)

        super().init()

    def save_model(self, path, global_step, *args, **kwargs):
        self.saver.save(sess=tf.get_default_session(),
                        save_path=path + '/' + self.name + '/',
                        global_step=global_step)
        super().save_model(path)

    def load_model(self, path, global_step, *args, **kwargs):
        import tensorflow as tf
        self.saver = tf.train.import_meta_graph(path + self.name + '/' + '-' + str(global_step) + '.meta')
        self.saver.recover_last_checkpoints(path + self.name + '/checkpoints')
        self.saver.restore(sess=tf.get_default_session(),
                           save_path=path + self.name + '/')
        super().load_model(path)

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            log = self.log_queue.get()
            print("%s: Loss %f: " % (self.name, log[self.name + '_LOSS']))
            log['INDEX'] = self.log_print_count
            self.log_file_content.append(log)
            self.log_print_count += 1
