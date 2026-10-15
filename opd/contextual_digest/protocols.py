def batched_handoff_implementation(protocol):
    if protocol != 'contextual_digest':
        raise ValueError('Only the complete Contextual Digest method is supported')
    from .digest import ContextualDigestBuilder, DIGEST_SYSTEM, DIGEST_PROTOCOL
    return ContextualDigestBuilder, DIGEST_SYSTEM, DIGEST_PROTOCOL
