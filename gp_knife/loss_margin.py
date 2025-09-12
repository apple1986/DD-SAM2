import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CELogitMarginL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)

    Args:
        margin (float, optional): The margin value. Defaults to 10.
        alpha (float, optional): The balancing weight. Defaults to 0.1.
        ignore_index (int, optional):
            Specifies a target value that is ignored
            during training. Defaults to -100.

        The following args are related to balancing weight (alpha) scheduling.
        Note all the results presented in our paper are obtained without the scheduling strategy.
        So it's fine to ignore if you don't want to try it.

        schedule (str, optional):
            Different stragety to schedule the balancing weight alpha or not:
            "" | add | multiply | step. Defaults to "" (no scheduling).
            To activate schedule, you should call function
            `schedula_alpha` every epoch in your training code.
        mu (float, optional): scheduling weight. Defaults to 0.
        max_alpha (float, optional): Defaults to 100.0.
        step_size (int, optional): The step size for updating alpha. Defaults to 100.
    """
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 class_weight = None,
                 ):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.class_weight = class_weight

        self.cross_entropy = nn.CrossEntropyLoss(self.class_weight)

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ce + self.alpha * loss_margin

        return loss #, loss_ce, loss_margin
    
########################abs distance as regularzation: limit the distance in a fixed range
class BCELogitMarginAbs(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 class_weight = None,
                 ):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.class_weight = class_weight

        # self.cross_entropy = nn.CrossEntropyLoss(self.class_weight)
        self.bin_cross_entropy = nn.BCEWithLogitsLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):
        loss_ce = self.bin_cross_entropy(inputs, targets)

        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(inputs - self.margin).mean()
        loss = loss_ce + self.alpha * loss_margin

        return loss

########################abs distance as regularzation: limit the distance in a fixed range, add sign regularization
class CELogitMarginAbs_SignReg(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2,
                 class_weight = None,
                 ):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes
        self.class_weight = class_weight

        self.cross_entropy = nn.CrossEntropyLoss(self.class_weight)

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()
        # loss_sign = (-10*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() * 0.001 ## notice: is it right? Yes. Make the labeled logit larger and the opposited smaller
        loss_sign = (-10*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() * 0.001
        loss_sign = torch.clip(loss_sign, -1, 1)
        loss = loss_ce + self.alpha * loss_margin + loss_sign

        # print("*"*40)
        # print(f"loss_ce: {loss_ce}, loss_lm: {self.alpha * loss_margin}, loss_sign: {loss_sign}")
        return loss

########################abs distance as regularzation: limit the distance in a fixed range, add sign regularization
class DiceLogitMarginAbs_SignReg(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.01,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.diceloss = DiceLoss(self.n_classes)

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_dice = self.diceloss(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()
        # loss_sign = (-10*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() * 0.001 ## notice: is it right? Yes. Make the labeled logit larger and the opposited smaller
        loss_sign = (-10*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() * 0.001
        loss_sign = torch.clip(loss_sign, -1, 1)
        loss = 10*loss_dice + self.alpha * loss_margin + loss_sign

        # print("*"*40)
        # print(f"loss_dice: {loss_dice}, loss_lm: {self.alpha * loss_margin}, loss_sign: {loss_sign}")
        return loss

## change the new name, same as DiceLogitMarginAbs_SignReg
class DiceLoss_MBFD(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.01,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.diceloss = DiceLoss(self.n_classes)

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_dice = self.diceloss(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()
        # loss_sign = (-10*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() * 0.001 ## notice: is it right? Yes. Make the labeled logit larger and the opposited smaller
        loss_sign = (-10*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() * 0.001
        loss_sign = torch.clip(loss_sign, -1, 1)
        loss = loss_dice + self.alpha * loss_margin + loss_sign

        # print("*"*40)
        # print(f"loss_dice: {loss_dice}, loss_lm: {self.alpha * loss_margin}, loss_sign: {loss_sign}")
        return loss

############################ Dice#+MarginLoss
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if inputs.size() != target.size():
            target = target.unsqueeze(1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


######################## fix the loss_sign
class CELogitMarginAbs_SignReg_fix(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()
        # ## make the corresponding labeled logit value larger and the oppiste value smaller
        # loss_obj =  inputs[:,1] * gt_hot[:, 1] # object
        # loss_bg = inputs[:,0] *  gt_hot[:, 0] # background
        # loss_sign = loss_obj.sum() / (gt_hot[:, 1]).sum() + loss_bg.sum() / (gt_hot[:, 0]).sum() 
        # loss_obj_opp = inputs[:,0] * (1 - gt_hot[:, 0]) # object oppsite
        # loss_bg_opp = inputs[:,1] *  (1-gt_hot[:, 1]) # background oppsite
        # loss_sign_opp = loss_obj_opp.sum() / (1-gt_hot[:, 0]).sum() + loss_bg_opp.sum() / (1-gt_hot[:, 1]).sum() 

        # loss = loss_ce + self.alpha * loss_margin - 0.001*loss_sign + 0.001*loss_sign_opp
        loss_sign = (-1*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() 
        loss_sign = torch.clip(loss_sign*0.1, -1, 1)
        loss = loss_ce + self.alpha * loss_margin + loss_sign
        print("*"*40)
        print(f"loss_ce: {loss_ce}, loss_lm: {self.alpha * loss_margin}, loss_sign: {loss_sign}")

        return loss




######################### learnable parameter
class CELogitMarginAbs_SignReg_Adap(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.learnable_param = nn.Parameter(torch.tensor(0.0005, requires_grad=True)).to("cuda:0")  # Initial value = 1.0
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()

        # loss_sign = (-10*(inputs*gt_hot) + (inputs*(1-gt_hot))).mean() ??
        loss_obj = -10 * inputs[:,1] * gt_hot[:, 1] # object
        loss_bg = inputs[:,0] *  gt_hot[:, 0] # background
        loss_sign = loss_obj.sum() / (gt_hot[:, 1]).sum() + loss_bg.sum() / (gt_hot[:, 0]).sum() 
        loss_sign_clip = torch.clip(self.learnable_param*loss_sign, -1, 1)
        
        # loss = loss_ce * 10 + self.alpha * loss_margin + loss_sign  # unet
        loss = loss_ce + self.alpha * loss_margin + loss_sign_clip #* self.learnable_param
               
        # print("*"*40)
        # print(f"learnable param: {self.learnable_param}")
        # print(f"loss_ce: {loss_ce}, loss_lm: {self.alpha * loss_margin}, loss_sign: {loss_sign * self.learnable_param}")

        return loss

######################### Absolute + sign regularization
class CELogitMarginAbs_Sign(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()
        ## get the sign between postive and negative logit value
        sign_judge_pos = (inputs[:,1] > inputs[:,0])* gt_hot[:, 1] # postive sample value greater thann negative sample value
        sign_judge_neg = (inputs[:,1] < inputs[:,0])* gt_hot[:, 0] #.to(inputs.dtype) 

        loss_sign = sign_judge_pos.sum() / gt_hot[:, 1].sum() + sign_judge_neg.sum() / gt_hot[:, 0].sum()
        loss =  loss_ce + self.alpha * loss_margin - loss_sign * 1
        print("*"*20)
        print(f"loss_ce: {loss_ce}, loss_lm: {self.alpha * loss_margin}, loss_sign: {loss_sign}")

        return loss


######################### Absolute + center cluster
class CELogitMarginAbs_Center(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()

        ## center of the positive/negative pixels
        ## the logit output has two channels, the 0th is negative, the 1st channel is positive
        pos_pixel = inputs[:, 1] * gt_hot[:, 1] # the value should be larger
        neg_pixel = inputs[:, 0] * gt_hot[:, 0] # the value should be larger
        pos_num = gt_hot[:, 1].sum()
        neg_num = gt_hot[:, 0].sum()
        pos_center = pos_pixel.sum() / pos_num
        neg_center = neg_pixel.sum() / neg_num
        ## for the oppisite direction
        pos_pixel_opp = inputs[:, 0] * (1-gt_hot[:, 0]) # the value should be smaller
        neg_pixel_opp = inputs[:, 1] * (1-gt_hot[:, 1]) # the value should be smaller
        pos_opp_num = pos_num # ((1-gt_hot[:, 0]).sum())
        neg_opp_num = neg_num # ((1-gt_hot[:, 1]).sum())
        pos_center_opp = pos_pixel_opp.sum() / pos_opp_num
        neg_center_opp = neg_pixel_opp.sum() / neg_opp_num

        ## the same pixels should be close
        loss_cluster = (torch.abs(pos_pixel - pos_center).sum() / pos_num  + torch.abs(neg_pixel - neg_center).sum() / neg_num  \
                            + torch.abs(pos_pixel_opp - pos_center_opp).sum() / pos_opp_num + torch.abs(neg_pixel_opp - neg_center_opp).sum() / neg_opp_num)
        ## the different pixels should be far, this can be done by loss_margin
        ## ce is for close the gt, margin contral the distance with the different labels, cluster is responsble for the same pixels
        loss = loss_ce + self.alpha * loss_margin + loss_cluster * 0.01
               
        print("*"*40)
        print(f"loss_ce: {loss_ce}, loss_lm: {self.alpha * loss_margin}, loss_sign: {loss_cluster * 0.01}")

        return loss

######################### Absolute + center cluster + variance the distance of different labels
class CELogitMarginAbs_Center_Dist(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = torch.abs(diff - self.margin).mean()

        ## center of the positive/negative pixels
        ## the logit output has two channels, the 0th is negative, the 1st channel is positive
        pos_pixel = inputs[:, 1] * gt_hot[:, 1] # the value should be larger
        neg_pixel = inputs[:, 0] * gt_hot[:, 0] # the value should be larger
        pos_num = gt_hot[:, 1].sum()
        neg_num = gt_hot[:, 0].sum()
        pos_center = pos_pixel.sum() / pos_num
        neg_center = neg_pixel.sum() / neg_num
        ## for the oppisite direction
        pos_pixel_opp = inputs[:, 0] * (1-gt_hot[:, 0]) # the value should be smaller
        neg_pixel_opp = inputs[:, 1] * (1-gt_hot[:, 1]) # the value should be smaller
        pos_opp_num = pos_num # ((1-gt_hot[:, 0]).sum())
        neg_opp_num = neg_num # ((1-gt_hot[:, 1]).sum())
        pos_center_opp = pos_pixel_opp.sum() / pos_opp_num
        neg_center_opp = neg_pixel_opp.sum() / neg_opp_num

        ## the same pixels should be close
        loss_cluster = (torch.abs(pos_pixel - pos_center).sum() / pos_num  + torch.abs(neg_pixel - neg_center).sum() / neg_num  \
                            + torch.abs(pos_pixel_opp - pos_center_opp).sum() / pos_opp_num + torch.abs(neg_pixel_opp - neg_center_opp).sum() / neg_opp_num)
        ## the different pixels should be far
        pos_dist = pos_pixel - pos_pixel_opp
        neg_dist = neg_pixel - neg_pixel_opp
        loss_dist = pos_dist.sum() / pos_num + neg_dist.sum() / neg_num

        ## ce is for close the gt, margin contral the distance with the different labels, cluster is responsble for the same pixels
        loss = loss_ce + self.alpha * loss_margin + loss_cluster * 0.001 - loss_dist * 0.001
               
        print("*"*40)
        print(f"loss_ce: {loss_ce}, loss_lm: {self.alpha * loss_margin}, loss_cluster: {loss_cluster * 0.1}, loss_dist: {loss_dist * 0.1}")

        return loss
    
######################### Absolute + center cluster + variance the distance of different labels
class CELogitMargin_Center_Dist(nn.Module):
    def __init__(self,
                 margin: float = 5,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 n_classes: int = 2):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size
        self.n_classes = n_classes

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, targets):
        # one-hot gt
        gt_hot = self._one_hot_encoder(targets.unsqueeze(1)) # N,C,H,W

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            # for gt
            gt_hot = gt_hot.view(gt_hot.size(0), gt_hot.size(1), -1)  # N,C,H,W => N,C,H*W
            gt_hot = gt_hot.transpose(1, 2)    # N,C,H*W => N,H*W,C
            gt_hot = gt_hot.contiguous().view(-1, gt_hot.size(2))   # N,H*W,C => N*H*W,C

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # # get logit distance
        # diff = self.get_diff(inputs)
        # # linear penalty where logit distances are larger than the margin
        # loss_margin = torch.abs(diff - self.margin).mean()

        ## center of the positive/negative pixels
        ## the logit output has two channels, the 0th is negative, the 1st channel is positive
        pos_pixel = inputs[:, 1] * gt_hot[:, 1] # the value should be larger
        neg_pixel = inputs[:, 0] * gt_hot[:, 0] # the value should be larger
        pos_num = gt_hot[:, 1].sum()
        neg_num = gt_hot[:, 0].sum()
        pos_center = pos_pixel.sum() / pos_num
        neg_center = neg_pixel.sum() / neg_num
        ## for the oppisite direction
        pos_pixel_opp = inputs[:, 0] * (1-gt_hot[:, 0]) # the value should be smaller
        neg_pixel_opp = inputs[:, 1] * (1-gt_hot[:, 1]) # the value should be smaller
        pos_opp_num = pos_num # ((1-gt_hot[:, 0]).sum())
        neg_opp_num = neg_num # ((1-gt_hot[:, 1]).sum())
        pos_center_opp = pos_pixel_opp.sum() / pos_opp_num
        neg_center_opp = neg_pixel_opp.sum() / neg_opp_num

        ## the same pixels should be close
        loss_cluster = (torch.abs(pos_pixel - pos_center).sum() / pos_num  + torch.abs(neg_pixel - neg_center).sum() / neg_num  \
                            + torch.abs(pos_pixel_opp - pos_center_opp).sum() / pos_opp_num + torch.abs(neg_pixel_opp - neg_center_opp).sum() / neg_opp_num)
        ## the different pixels should be far
        pos_dist = pos_pixel - pos_pixel_opp
        neg_dist = neg_pixel - neg_pixel_opp
        loss_dist = pos_dist.sum() / pos_num + neg_dist.sum() / neg_num

        ## ce is for close the gt, margin contral the distance with the different labels, cluster is responsble for the same pixels
        loss = loss_ce + self.alpha * loss_cluster - loss_dist * 0.01
               
        print("*"*40)
        print(f"loss_ce: {loss_ce}, loss_cluster: {self.alpha * loss_cluster}, loss_dist: {loss_dist * 0.001}")

        return loss
    
##########################
class LogitMarginBoth(CELogitMarginL1):
    def __init__(self,
                 margin: float = 10,
                 margin_low: float = 1.0,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.margin_low = margin_low
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()

    
    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin_high = F.relu(diff-self.margin).mean()
        loss_margin_low = torch.clip(diff, 0, self.margin_low).mean()
        loss = loss_ce + self.alpha * loss_margin_high - self.alpha * loss_margin_low

        return loss
########################### result: it is not the best
class CELogitMarginBothL2(CELogitMarginL1):
    def __init__(self,
                 margin: float = 10,
                 margin_low: float = 1.0,
                 alpha: float = 0.0001,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.margin_low = margin_low
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()

    
    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        reg_diff = (diff - self.margin)**2
        # give penalty where logit distances are larger than the margin
        loss = loss_ce + self.alpha * reg_diff.mean()
        return loss

########################### for pd calibration: result: it is not the best
class CEpdMarginBoth(CELogitMarginL1):
    def __init__(self,
                 margin: float = 10,
                 margin_low: float = 1.0,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.margin_low = margin_low
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()

    
    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        ## softmax the logit
        inputs = torch.softmax(inputs, dim=1)

        # get pd distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin_high = F.relu(diff-self.margin).mean()
        loss_margin_low = torch.clip(diff, 0, self.margin_low).mean()
        loss = loss_ce + self.alpha * loss_margin_high - self.alpha * loss_margin_low

        return loss


############################
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # if inputs.size() != target.size():
        if len(inputs.shape) != len(target.shape):
            target = target.unsqueeze(1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class DiceLossMargBoth(nn.Module):
    def __init__(self, n_classes, margin_low=1, margin=5, alpha=0.1, both=False):
        super(DiceLossMargBoth, self).__init__()
        self.n_classes = n_classes
        self.margin_low = margin_low
        self.margin = margin
        self.alpha = alpha
        self.both = both

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff
    
    def get_dice(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        return loss / self.n_classes


    def forward(self, inputs, targets, weight=None, softmax=False):
        loss_dice = self.get_dice(inputs, targets, weight=None, softmax=softmax)

        ## cal margin
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin_high = F.relu(diff-self.margin).mean()
        if self.both:
            loss_margin_low = torch.clip(diff, 0, self.margin_low).mean()
            loss = loss_dice + self.alpha * loss_margin_high + self.alpha * (-loss_margin_low)
        else:
            loss = loss_dice + self.alpha * loss_margin_high


        return loss
    

########## label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, ignore_index=-100, reduction="mean"):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(target != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index]

        confidence = 1. - self.alpha
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.alpha * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        

##################### ECP
class EntropyConfidencePenalty(nn.Module):
    """Regularizing neural networks by penalizing confident output distributions, 2017. <https://arxiv.org/pdf/1701.06548>

        loss = CE - alpha * Entropy(p)
    """
    def __init__(self, alpha=1.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

    @property
    def names(self):
        return "loss", "loss_ce", "loss_ent"

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        # cross entropy
        loss_ce = F.cross_entropy(inputs, targets)

        # entropy
        prob = F.log_softmax(inputs, dim=1).exp()
        prob = torch.clamp(prob, 1e-10, 1.0 - 1e-10)
        ent = - prob * torch.log(prob)
        loss_ent = ent.mean()

        loss = loss_ce - self.alpha * loss_ent

        return loss#, loss_ce, loss_ent
    
################SVLS
def get_gaussian_kernel_2d(ksize=0, sigma=0):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp( 
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16)
        )
    return gaussian_kernel / torch.sum(gaussian_kernel)

class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_svls_filter_2d, self).__init__()
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
        neighbors_sum = (1 - gkernel[1,1]) + 1e-16
        gkernel[int(ksize/2), int(ksize/2)] = neighbors_sum
        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = int(ksize/2)
        self.svls_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding, padding_mode='replicate')
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False
    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()

class CELossWithSVLS_2D(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, ksize=3):
        super(CELossWithSVLS_2D, self).__init__()
        self.cls = torch.tensor(classes)
        self.svls_layer = get_svls_filter_2d(ksize=3, sigma=sigma, channels=self.cls)
        self.svls_layer = self.svls_layer.to("cuda:1")

    def forward(self, inputs, labels):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()
        svls_labels = self.svls_layer(oh_labels)
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()