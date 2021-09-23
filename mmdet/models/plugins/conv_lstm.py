import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size,
                 bias=True, relu=True, dilation=1,
                 dropout_data=0.5, dropout_recurrent=0.5):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dilation = dilation

        self.kernel_size = kernel_size
        self.padding = ((dilation * (kernel_size[0] - 1)) // 2, (dilation * (kernel_size[1] - 1)) // 2)
        self.bias = bias
        self.dropout_data = dropout_data
        self.dropout_recurrent = dropout_recurrent
        self.activity = F.relu if relu else F.tanh

        self.conv_x_i = self._make_conv_data()
        self.conv_x_f = self._make_conv_data()
        self.conv_x_o = self._make_conv_data()
        self.conv_x_g = self._make_conv_data()

        self.conv_h_i = self._make_conv_recurrent()
        self.conv_h_f = self._make_conv_recurrent()
        self.conv_h_o = self._make_conv_recurrent()
        self.conv_h_g = self._make_conv_recurrent()

        self.dropout_x_i = nn.Dropout2d(self.dropout_data)
        self.dropout_x_f = nn.Dropout2d(self.dropout_data)
        self.dropout_x_o = nn.Dropout2d(self.dropout_data)
        self.dropout_x_g = nn.Dropout2d(self.dropout_data)

        self.dropout_h_i = nn.Dropout2d(self.dropout_recurrent)
        self.dropout_h_f = nn.Dropout2d(self.dropout_recurrent)
        self.dropout_h_o = nn.Dropout2d(self.dropout_recurrent)
        self.dropout_h_g = nn.Dropout2d(self.dropout_recurrent)

        self._init_parameters()

    def _make_conv_data(self):
        conv_data = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=False,
                              dilation=self.dilation)
        return conv_data

    def _make_conv_recurrent(self):
        conv_recurrent = nn.Conv2d(in_channels=self.hidden_dim,
                                   out_channels=self.hidden_dim,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   bias=True,
                                   dilation=self.dilation)
        return conv_recurrent

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state

        x_i = self.conv_x_i(self.dropout_x_i(x))
        x_f = self.conv_x_f(self.dropout_x_f(x))
        x_o = self.conv_x_o(self.dropout_x_o(x))
        x_g = self.conv_x_g(self.dropout_x_g(x))

        h_i = self.conv_h_i(self.dropout_h_i(h_cur))
        h_f = self.conv_h_f(self.dropout_h_f(h_cur))
        h_o = self.conv_h_o(self.dropout_h_o(h_cur))
        h_g = self.conv_h_g(self.dropout_h_g(h_cur))

        i = torch.sigmoid(x_i + h_i)
        f = torch.sigmoid(x_f + h_f)
        o = torch.sigmoid(x_o + h_o)
        g = self.activity(x_g + h_g)

        c_next = f * c_cur + i * g
        h_next = o * self.activity(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        h = nn.Parameter(torch.zeros((batch_size, self.hidden_dim, height, width), device=torch.device("cuda")))
        c = nn.Parameter(torch.zeros((batch_size, self.hidden_dim, height, width), device=torch.device("cuda")))
        return (h, c)

    def _init_parameters(self):

        nn.init.xavier_uniform_(self.conv_x_i.weight)
        nn.init.xavier_uniform_(self.conv_x_f.weight)
        nn.init.xavier_uniform_(self.conv_x_o.weight)
        nn.init.xavier_uniform_(self.conv_x_g.weight)

        nn.init.orthogonal_(self.conv_h_i.weight)
        nn.init.orthogonal_(self.conv_h_f.weight)
        nn.init.orthogonal_(self.conv_h_o.weight)
        nn.init.orthogonal_(self.conv_h_g.weight)

        nn.init.constant_(self.conv_h_i.bias, 0)
        nn.init.constant_(self.conv_h_f.bias, 1)
        nn.init.constant_(self.conv_h_o.bias, 0)
        nn.init.constant_(self.conv_h_g.bias, 0)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1,
                 batch_first=True, bias=True, birnn=False, relu=True,
                 dilation=1, dropout_data=0, dropout_recurrent=0):
        super(ConvLSTM, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.bias = bias
        self.birnn = birnn
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cell_list_pos = nn.ModuleList()
        if self.birnn:
            self.reduce_conv = nn.Conv2d(in_channels=self.hidden_dim[-1] * 2,
                                         out_channels=self.hidden_dim[-1],
                                         kernel_size=1)
            self.cell_list_neg = nn.ModuleList()

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            self.cell_list_pos.append(ConvLSTMCell(input_dim=cur_input_dim,
                                                   hidden_dim=self.hidden_dim[i],
                                                   kernel_size=self.kernel_size[i],
                                                   bias=self.bias, relu=relu,
                                                   dilation=dilation,
                                                   dropout_data=dropout_data,
                                                   dropout_recurrent=dropout_recurrent))
            if self.birnn:
                self.cell_list_neg.append(ConvLSTMCell(input_dim=cur_input_dim,
                                                       hidden_dim=self.hidden_dim[i],
                                                       kernel_size=self.kernel_size[i],
                                                       bias=self.bias, relu=relu,
                                                       dilation=dilation,
                                                       dropout_data=dropout_data,
                                                       dropout_recurrent=dropout_recurrent))
        self.to(torch.device("cuda"))
    def forward(self, input_tensor, test_mode=False, is_first=False, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        seq_len = input_tensor.size(1)
        middle = int(seq_len / 2)

        # Forward process
        cur_layer_input_pos = input_tensor
        input_shape = input_tensor.size()
        hidden_state_pos = self._init_hidden(input_shape[0], input_shape[-2], input_shape[-1])
        #import pdb;pdb.set_trace()
        # add test mode
        if test_mode and not is_first:
           hidden_state_pos = [hidden_state]

        for layer_idx in range(self.num_layers):

            output_inner = []
            h, c = hidden_state_pos[layer_idx]
            for t in range(seq_len):
                h, c = self.cell_list_pos[layer_idx](x=cur_layer_input_pos[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input_pos = layer_output

        layer_output_list_pos = layer_output

        # Backward process
        if self.birnn:
            cur_layer_input_neg = input_tensor
            hidden_state_neg = self._init_hidden(input_shape[0], input_shape[-2], input_shape[-1])

            for layer_idx in range(self.num_layers):

                output_inner = []
                h, c = hidden_state_neg[layer_idx]
                for t in range(seq_len - 1, -1, -1):
                    h, c = self.cell_list_neg[layer_idx](x=cur_layer_input_neg[:, t, :, :, :], cur_state=[h, c])
                    output_inner.append(h)

                output_inner.reverse()
                layer_output = torch.stack(output_inner, dim=1)
                cur_layer_input_neg = layer_output

            layer_output_list_neg = layer_output
            # Combine BiDirection
            lstm_output = torch.cat((layer_output_list_pos, layer_output_list_neg), dim=2)
            lstm_output = lstm_output.reshape(-1, self.hidden_dim[-1] * 2, input_shape[-2], input_shape[-1])
            lstm_output = F.relu(self.reduce_conv(lstm_output))
            lstm_output = lstm_output.reshape(-1, seq_len, self.hidden_dim[-1], input_shape[-2], input_shape[-1])
        else:
            lstm_output = layer_output_list_pos
        if test_mode:
           return lstm_output[:, -1], [h, c] 
        else: 
           return lstm_output[:, -1]

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list_pos[i].init_hidden(batch_size, height, width))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



