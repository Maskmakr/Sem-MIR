"""Vectorized flat-fading MMSE-PIC, positive-one bit LLR convention."""
import numpy as np

def make_constellation(n_tx=4):
    labels = ((np.arange(16)[:,None] >> np.arange(4)) & 1).astype(np.uint8)
    levels = np.array([-3,-1,3,1])
    symbols = (levels[:,None] + 1j*levels[None,:]).ravel()
    # Share unit nominal transmit-vector energy across the antenna streams.
    symbols = symbols / np.sqrt(n_tx * np.mean(np.abs(symbols)**2))
    return symbols, labels

def mmse_pic(y, h, noise_variance, posterior_llr):
    """Return (...,n_tx,4) detector LLRs for y shaped (...,n_rx)."""
    y = np.asarray(y, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    nt = h.shape[-1]
    nr = h.shape[-2]
    n0 = float(noise_variance)
    posterior_llr = np.asarray(posterior_llr, dtype=np.float64)
    constellation, labels = make_constellation(nt)
    # Independent bit APPs define symbol probabilities; positive LLR favors one.
    signs = 2.0 * labels - 1.0
    log_probability = 0.5 * np.einsum('...tb,qb->...tq',posterior_llr, signs)
    log_probability -= log_probability.max(axis=-1, keepdims=True)
    probability = np.exp(log_probability)
    probability /= probability.sum(axis=-1, keepdims=True)
    mean = np.sum(probability*constellation, axis=-1)
    variance = np.sum(probability*np.abs(constellation-mean[...,None])**2, axis=-1)
    hh = h.conj().swapaxes(-1,-2)
    gram = hh @ h
    # With V=diag(variance), solve (H^H H V + N0 I) G = H^H.
    g = np.linalg.solve(gram * variance[...,None,:] + n0*np.eye(nt), np.broadcast_to(hh,variance.shape[:-1]+(nt,nr)))
    gh = g @ h
    # Cancel other streams' soft means while retaining the target contribution.
    cancelled = y - np.einsum('rt,...t->...r',h,mean)
    output = np.einsum('...tr,...r->...t',g,cancelled) + np.diagonal(gh,axis1=-2,axis2=-1)*mean
    gain = np.diagonal(gh,axis1=-2,axis2=-1)
    # Residual interference plus filtered complex noise sets the metric scale.
    npi = np.sum(np.abs(gh)**2*variance[...,None,:],axis=-1) - np.abs(gain)**2*variance + n0*np.sum(np.abs(g)**2,axis=-1)
    metrics = np.abs(output[...,None]-gain[...,None]*constellation)**2 / np.maximum(npi[...,None],np.finfo(float).tiny)
    llrs = np.empty(posterior_llr.shape)
    # Zero-minus-one distance returns positive-one max-log LLRs.
    for bit in range(4):
        llrs[...,bit] = metrics[...,labels[:,bit]==0].min(axis=-1) - metrics[...,labels[:,bit]==1].min(axis=-1)
    return llrs
