import hydra
import torch
import json
import os
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from lyra_lite.utils.seeds import set_seed
from lyra_lite.training.trainer import train_one_epoch, evaluate
import logging
from lyra_lite.data.dataset import CellDataset
log = logging.getLogger(__name__)



@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    X, y, patient_id = hydra.utils.instantiate(cfg.data)

    # patient-level split: 70% train, 15% val, 15% test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=cfg.seed)
    train_idx, temp_idx = next(gss1.split(X, y, groups=patient_id))

    X_train, y_train, patient_id_train = X[train_idx], y[train_idx], patient_id[train_idx]
    X_temp, y_temp, patient_id_temp = X[temp_idx], y[temp_idx], patient_id[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=cfg.seed)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=patient_id_temp))

    X_val, y_val, patient_id_val = X_temp[val_idx], y_temp[val_idx], patient_id_temp[val_idx]
    X_test, y_test, patient_id_test = X_temp[test_idx], y_temp[test_idx], patient_id_temp[test_idx]


    log.info(f"(Pos Rate) - Train: {y_train.mean():.2f} | Val: {y_val.mean():.2f} | Test: {y_test.mean():.2f}")

    # importance ordering: fit a probe on the train split only
    ordering = cfg.representation.get("ordering", "rank")
    gene_importance = None
    if str(ordering).startswith("importance"):
        from lyra_lite.data.representation import compute_gene_importance
        gene_importance = compute_gene_importance(X_train, y_train, seed=cfg.seed)
        log.info(f"Fitted gene-importance probe for ordering='{ordering}' (n_genes={gene_importance.shape[0]}).")

    # datasets + dataloaders
    train_dataset = CellDataset(
        X=X_train,
        y=y_train,
        representation=cfg.representation.representation,
        top_k=cfg.representation.get("top_k", 256),
        ordering=cfg.representation.get("ordering", "rank"),
        use_value=cfg.representation.get("use_value", True),
        pad_id=cfg.representation.get("pad_id", None),
        importance=gene_importance
    )
    val_dataset = CellDataset(
        X=X_val,
        y=y_val,
        representation=cfg.representation.representation,
        top_k=cfg.representation.get("top_k", 256),
        ordering=cfg.representation.get("ordering", "rank"),
        use_value=cfg.representation.get("use_value", True),
        pad_id=cfg.representation.get("pad_id", None),
        importance=gene_importance
    )
    test_dataset = CellDataset(
        X=X_test,
        y=y_test,
        representation=cfg.representation.representation,
        top_k=cfg.representation.get("top_k", 256),
        ordering=cfg.representation.get("ordering", "rank"),
        use_value=cfg.representation.get("use_value", True),
        pad_id=cfg.representation.get("pad_id", None),
        importance=gene_importance
    )

    num_workers = cfg.training.num_workers

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers)

    # model + optimizer + loss
    device = torch.device(cfg.device)
    model = hydra.utils.instantiate(cfg.model).to(device)

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.training.loss)

    # training loop with early stopping
    best_val_loss = float("inf")
    best_auroc = float("nan")
    best_ece = float("nan")
    best_epoch = -1
    patience_counter = 0
    best_model_weights = None

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device )
        val_loss, metrics = evaluate(model, val_loader, criterion, device, n_bins=cfg.training.metrics.ece_n_bins)
        log.info(f"Epoch {epoch+1}/{cfg.training.epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | AUROC: {metrics['auroc']:.4f} | ECE: {metrics['ece']:.4f}")
        if val_loss < best_val_loss - cfg.training.early_stopping.min_delta:
            best_val_loss = val_loss
            best_auroc = metrics['auroc']
            best_auprc = metrics['auprc']
            best_ece = metrics['ece']
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        if patience_counter >= cfg.training.early_stopping.patience:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    # evaluate the best model on the test set
    log.info("--- training completed, evaluating on test set ---")
    model.load_state_dict(best_model_weights)

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        n_bins=cfg.training.metrics.ece_n_bins)

    log.info(f"FINAL TEST RESULTS | test_loss: {test_loss:.4f} | test_AUROC: {test_metrics['auroc']:.4f} | test_ECE: {test_metrics['ece']:.4f}")

    # save metrics + best weights
    out_dir = HydraConfig.get().runtime.output_dir
    metrics = {
        "seed": cfg.seed,
        "model": cfg.model._target_,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_auroc": float(best_auroc),
        "val_auprc": float(best_auprc),
        "val_ece": float(best_ece),
        "test_loss": float(test_loss),
        "test_auroc": float(test_metrics['auroc']),
        "test_auprc": float(test_metrics['auprc']),
        "test_ece": float(test_metrics['ece'])
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {out_dir}/metrics.json")

    torch.save(best_model_weights, os.path.join(out_dir, "best_model.pt"))

if __name__ == "__main__":
    main()
