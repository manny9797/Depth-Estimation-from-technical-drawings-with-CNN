import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).cuda()

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        mask = (diff <= self.threshold).float()

        l1_part = mask * diff
        l2_part = (1 - mask) * (diff * diff + self.threshold * self.threshold) / (2 * self.threshold)

        return torch.mean(l1_part + l2_part)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size, channel):
        def _gaussian(window_size, sigma):
            x = torch.arange(window_size, dtype=torch.float32, device='cpu')  # Usa un tensore direttamente
            gauss = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
            return gauss / gauss.sum()


        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.device == img1.device:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device)
            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.3, berhu_weight=0.4, gradient_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.ssim_loss = SSIM()
        self.berhu_loss = BerHuLoss()
        self.gradient_loss = GradientLoss()
        self.ssim_weight = ssim_weight
        self.berhu_weight = berhu_weight
        self.gradient_weight = gradient_weight

    def forward(self, pred, target):
        ssim = self.ssim_loss(pred, target)
        berhu = self.berhu_loss(pred, target)
        gradient = self.gradient_loss(pred, target)

        total_loss = (self.ssim_weight * ssim +
                      self.berhu_weight * berhu +
                      self.gradient_weight * gradient)

        # Restituisci anche le singole loss
        return total_loss, {'ssim': ssim.item(), 'berhu': berhu.item(), 'gradient': gradient.item()}