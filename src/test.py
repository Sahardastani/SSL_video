import torch

def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()                                                 #torch.Size([101, 9537])
    num_test_images, num_chunks = test_labels.shape[0], 100                             #3783, 100
    imgs_per_chunk = num_test_images // num_chunks                                      #37
    retrieval_one_hot = torch.zeros(k, num_classes)                                     #torch.Size([5, 101])

    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[                                                       #torch.Size([37, 101])
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]       #torch.Size([37])
        batch_size = targets.shape[0]
        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)                                 #torch.Size([37, 9537])
        distances, indices = similarity.topk(k, largest=True, sorted=True)              #torch.Size([37, 5])
        candidates = train_labels.view(1, -1).expand(batch_size, -1)                    #torch.Size([37, 9537])
        retrieved_neighbors = torch.gather(candidates, 1, indices).to(torch.int64)      #torch.Size([37, 5])

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()                  #torch.Size([185, 101])
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

all_train = torch.randn((9537, 400))
all_val = torch.randn((3783, 400))

train_labels = torch.randn(9537)
test_labels = torch.randn(3783)

top1, top5 = knn_classifier(all_train, train_labels, all_val, test_labels, k = 5, T = 0.07)
print(top1, top5)