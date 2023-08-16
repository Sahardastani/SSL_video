import torch

class VideoDataAugmentationDINO(object):
    # ... (rest of the class remains the same) ...

    def __call__(self, image, from_list=False, no_aug=False, two_token=False):
        if two_token:
            image = [x.float() / 255.0 if x.dtype == torch.uint8 else x for x in image]
            crops = [self.global_transform1(image[0]), self.no_aug(image[0]),
                     self.local_transform(image[1]), self.local_transform(image[2]),
                     self.no_aug(image[3]), self.no_aug(image[4])]
        elif no_aug:
            image = [x.float() / 255.0 if x.dtype == torch.uint8 else x for x in image]
            crops = [self.no_aug(x) for x in image]
        elif from_list:
            image = [x.float() / 255.0 if x.dtype == torch.uint8 else x for x in image]
            crops = [self.global_transform1(image[0]), self.global_transform2(image[1])]
            for local_image in image[2:]:
                crops.append(self.local_transform(local_image))
        else:
            if image.dtype == torch.uint8:
                image = image.float()
                image = image / 255.0
            crops = [self.global_transform1(image), self.global_transform2(image)]
            for _ in range(self.local_crops_number):
                crops.append(self.local_transform(image))

        # Create the tensor representing the frame locations for each clip
        batch_size = crops[0].shape[0]
        num_frames = len(crops)
        frame_locations = torch.arange(0, num_frames).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

        return crops, frame_locations
