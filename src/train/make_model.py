from src.model.mtgru import multiGRU
from src.model.mtgru_overrall import multiGRU_OverRall
from src.model.mtcapsule import multiCapsule
from src.model.mtcapsule_overrall import multiCapsule_Overrall
from src.model.mtcapsule_overrall_variant import multiCapsule_Overrall_Variant
from src.model.mtcapsule_overrall_routing import multiCapsule_Overrall_Routing
from src.model.mtcapsule_label_routing import multiCapsule_Label_Routing


def make_model(config, word_embedding):
    model_type = config.model_type
    if model_type == 'han_model':
        model = multiGRU(
            max_sen_len=config.max_sentence_len,
            n_class_1=config.n_class_1,
            n_class_2=config.n_class_2,
            n_class_3=config.n_class_3,
            num_heads=config.num_heads,
            embedding_document=word_embedding,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            random_base=config.random_base,
            l2_reg=config.l2_reg
        )
    elif model_type == 'han_model_overrall':
        model = multiGRU_OverRall(
            max_sen_len=config.max_sentence_len,
            n_class_1=config.n_class_1,
            n_class_2=config.n_class_2,
            n_class_3=config.n_class_3,
            num_heads=config.num_heads,
            embedding_document=word_embedding,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            random_base=config.random_base,
            l2_reg=config.l2_reg
        )
    elif model_type == 'HAN_capsule':
        model = multiCapsule(
            max_sen_len=config.max_sentence_len,
            n_class_1=config.n_class_1,
            n_class_2=config.n_class_2,
            n_class_3=config.n_class_3,
            embedding_document=word_embedding,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            sc_num=config.sc_num,
            sc_dim=config.sc_dim,
            cc_dim=config.cc_dim,
            filter_size=config.filter_size,
            iter_routing=config.iter_routing,
            random_base=config.random_base,
            l2_reg=config.l2_reg
        )
    elif model_type == 'HAN_capsule_overrall':
        model = multiCapsule_Overrall(
            max_sen_len=config.max_sentence_len,
            n_class_1=config.n_class_1,
            n_class_2=config.n_class_2,
            n_class_3=config.n_class_3,
            embedding_document=word_embedding,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            sc_num=config.sc_num,
            sc_dim=config.sc_dim,
            cc_dim=config.cc_dim,
            filter_size=config.filter_size,
            iter_routing=config.iter_routing,
            random_base=config.random_base,
            l2_reg=config.l2_reg
        )
    elif model_type == 'HAN_capsule_overrall_variant':
        model = multiCapsule_Overrall_Variant(
            max_sen_len=config.max_sentence_len,
            n_class_1=config.n_class_1,
            n_class_2=config.n_class_2,
            n_class_3=config.n_class_3,
            embedding_document=word_embedding,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            sc_num=config.sc_num,
            sc_dim=config.sc_dim,
            cc_dim=config.cc_dim,
            filter_size=config.filter_size,
            iter_routing=config.iter_routing,
            random_base=config.random_base,
            l2_reg=config.l2_reg
        )
    elif model_type == 'multiCapsule_Overrall_Routing':
        model = multiCapsule_Overrall_Routing(
            max_sen_len=config.max_sentence_len,
            n_class_1=config.n_class_1,
            n_class_2=config.n_class_2,
            n_class_3=config.n_class_3,
            embedding_document=word_embedding,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            sc_num=config.sc_num,
            sc_dim=config.sc_dim,
            cc_dim=config.cc_dim,
            filter_size=config.filter_size,
            iter_routing=config.iter_routing,
            random_base=config.random_base,
            l2_reg=config.l2_reg
        )
    elif model_type == 'multiCapsule_Label_Routing':
        model = multiCapsule_Label_Routing(
            max_sen_len=config.max_sentence_len,
            n_class_1=config.n_class_1,
            n_class_2=config.n_class_2,
            n_class_3=config.n_class_3,
            embedding_document=word_embedding,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            sc_num=config.sc_num,
            sc_dim=config.sc_dim,
            cc_dim=config.cc_dim,
            filter_size=config.filter_size,
            iter_routing=config.iter_routing,
            random_base=config.random_base,
            l2_reg=config.l2_reg
        )
    else:
        raise ValueError('model error:', model_type)
    return model