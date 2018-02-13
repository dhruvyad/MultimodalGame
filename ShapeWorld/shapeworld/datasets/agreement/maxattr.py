from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import ReinforcedAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, MaxAttributeCaptioner, ExistentialCaptioner


class Maxattr(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=14,
        vocabulary=('.', 'a', 'an', 'biggest', 'blue', 'circle', 'cross', 'cyan', 'darkest', 'ellipse', 'gray', 'green', 'is', 'leftmost', 'lightest', 'lowermost', 'magenta', 'most', 'pentagon', 'rectangle', 'red', 'rightmost', 'semicircle', 'shape', 'smallest', 'square', 'topmost', 'triangle', 'yellow'),
        language=None
    ):

        world_generator = ReinforcedAttributesGenerator(
            reinforcement_range=(1, 1),
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )

        world_captioner = ExistentialCaptioner(
            restrictor_captioner=MaxAttributeCaptioner(
                scope_captioner=RegularTypeCaptioner(
                    hypernym_rate=1.0,
                    logical_tautology_rate=1.0
                )
            ),
            body_captioner=AttributeTypeRelationCaptioner(
                attribute_type_captioner=CaptionerMixer(
                    captioners=(
                        RegularAttributeCaptioner(),
                        RegularTypeCaptioner()
                    )
                )
            )
        )

        super(Maxattr, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = Maxattr
