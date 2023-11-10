    # @staticmethod
    # def correlation_metric(x):
    #     x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
    #     return torch.mean(
    #         model_utils.off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
    #     )
    
    # @staticmethod
    # def std_metric(x):
    #         x = F.normalize(x, p=2, dim=1)
    #         return torch.mean(x.std(dim=0))
    
    # def compute_metrics(self, outputs, is_val= False):

    #     representation = model_utils.batch_all_gather(outputs["representation"][0])
    #     corr = self.correlation_metric(representation)
    #     stdrepr = self.std_metric(representation)

    #     if self.cfg.MODEL.ALPHA > 0.0:
    #         embedding = model_utils.batch_all_gather(outputs["embedding"][0])
    #         core = self.correlation_metric(embedding)
    #         stdemb = self.std_metric(embedding)
    #         if is_val:
    #             self.log('eval_stdr', stdrepr)
    #             self.log('eval_stde', stdemb)
    #             self.log('eval_corr', corr)
    #             self.log('eval_core', core)
    #         else:
    #             self.log('train_stdr', stdrepr)
    #             self.log('train_stde', stdemb)
    #             self.log('train_corr', corr)
    #             self.log('train_core', core)