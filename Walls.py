import pygame


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, win):
        pygame.draw.line(win, (255, 255, 255), (self.x1, self.y1), (self.x2, self.y2), 5)


def getWalls():
    walls = []

    wall1 = Wall(32, 160, 32, 460)
    wall2 = Wall(32, 460, 40, 512)
    wall3 = Wall(40, 512, 61, 547)
    wall4 = Wall(61, 547, 98, 578)
    wall5 = Wall(98, 578, 151, 598)
    wall6 = Wall(151, 598, 172, 600)
    wall7 = Wall(172, 600, 227, 599)  #
    wall8 = Wall(227, 599, 229, 557)
    wall9 = Wall(229, 557, 340, 557)
    wall10 = Wall(340, 557, 340, 599)
    wall11 = Wall(340, 599, 671, 599)
    wall12 = Wall(671, 599, 670, 568)  #
    wall13 = Wall(670, 568, 697, 565)
    wall14 = Wall(697, 565, 709, 557)
    wall15 = Wall(709, 557, 725, 556)
    wall16 = Wall(725, 556, 746, 567)
    wall17 = Wall(746, 567, 757, 571)
    wall18 = Wall(757, 571, 758, 598)
    wall19 = Wall(758, 598, 776, 594)
    wall20 = Wall(776, 594, 814, 590)
    wall21 = Wall(814, 590, 878, 559)
    wall22 = Wall(878, 559, 913, 524)
    wall23 = Wall(913, 524, 933, 483)
    wall24 = Wall(933, 483, 931, 434)
    wall25 = Wall(931, 434, 906, 385)
    wall27 = Wall(906, 385, 865, 351)
    wall28 = Wall(865, 351, 808, 328)
    wall29 = Wall(808, 328, 721, 325)
    wall30 = Wall(721, 325, 723, 363)
    wall31 = Wall(723, 363, 638, 362)
    wall32 = Wall(638, 362, 637, 339)
    wall33 = Wall(637, 339, 636, 323)
    wall34 = Wall(636, 323, 365, 324)  #
    wall35 = Wall(365, 324, 339, 311)
    wall36 = Wall(339, 311, 333, 288)
    wall37 = Wall(333, 288, 347, 266)
    wall38 = Wall(347, 266, 365, 257)
    wall39 = Wall(365, 257, 432, 257)
    wall40 = Wall(432, 257, 433, 227)
    wall41 = Wall(433, 227, 464, 225)
    wall42 = Wall(464, 225, 486, 211)  #
    wall43 = Wall(486, 211, 510, 222)  #
    wall44 = Wall(510, 222, 530, 234)
    wall45 = Wall(530, 234, 531, 257)
    wall46 = Wall(531, 257, 750, 255)
    wall47 = Wall(750, 255, 814, 239)
    wall48 = Wall(814, 239, 854, 209)
    wall49 = Wall(814, 239, 854, 209)
    wall50 = Wall(854, 209, 885, 160)
    wall51 = Wall(885, 160, 889, 111)
    wall52 = Wall(889, 111, 860, 57)
    wall53 = Wall(860, 57, 800, 27)
    wall54 = Wall(800, 27, 698, 24)
    wall55 = Wall(698, 24, 699, 67)
    wall56 = Wall(699, 67, 584, 66)
    wall57 = Wall(584, 66, 583, 21)
    wall58 = Wall(583, 21, 175, 20)
    wall59 = Wall(175, 20, 114, 36)  #
    wall60 = Wall(114, 36, 70, 63)
    wall61 = Wall(70, 63, 40, 109)
    wall62 = Wall(40, 109, 32, 160)
    wall63 = Wall(135, 154, 140, 475)
    wall64 = Wall(140, 475, 180, 494)
    wall65 = Wall(180, 494, 331, 496)
    wall66 = Wall(331, 496, 455, 496)
    wall67 = Wall(455, 496, 456, 537)
    wall68 = Wall(456, 537, 573, 539)
    wall69 = Wall(573, 539, 575, 496)
    wall70 = Wall(575, 496, 766, 494)
    wall71 = Wall(766, 494, 797, 470)
    wall72 = Wall(797, 470, 791, 442)
    wall73 = Wall(791, 442, 768, 428)
    wall74 = Wall(768, 428, 523, 427)
    wall75 = Wall(523, 427, 520, 399)
    wall76 = Wall(520, 399, 483, 382)
    wall77 = Wall(483, 382, 432, 395)
    wall78 = Wall(432, 395, 427, 423)
    wall79 = Wall(427, 423, 364, 427)
    wall80 = Wall(364, 427, 311, 418)
    wall81 = Wall(311, 418, 231, 339)
    wall82 = Wall(231, 339, 229, 249)
    wall83 = Wall(229, 249, 268, 192)
    wall84 = Wall(268, 192, 349, 148)
    wall85 = Wall(349, 148, 490, 155)  #
    wall86 = Wall(490, 155, 612, 152)
    wall87 = Wall(612, 152, 610, 194)
    wall88 = Wall(610, 194, 696, 198)
    wall89 = Wall(696, 198, 700, 151)
    wall90 = Wall(700, 151, 765, 151)
    wall91 = Wall(765, 151, 785, 126)
    wall92 = Wall(785, 126, 371, 127)
    wall93 = Wall(371, 127, 367, 83)
    wall94 = Wall(367, 83, 253, 85)
    wall95 = Wall(253, 85, 251, 126)  #
    wall96 = Wall(251, 126, 155, 128)
    wall97 = Wall(155, 128, 135, 154)
    wall98 = Wall(135, 154, 135, 154)

    walls.append(wall1)
    walls.append(wall2)
    walls.append(wall3)
    walls.append(wall4)
    walls.append(wall5)
    walls.append(wall6)
    walls.append(wall7)
    walls.append(wall8)
    walls.append(wall9)
    walls.append(wall10)
    walls.append(wall11)
    walls.append(wall12)
    walls.append(wall13)
    walls.append(wall14)
    walls.append(wall15)
    walls.append(wall16)
    walls.append(wall17)
    walls.append(wall18)
    walls.append(wall19)
    walls.append(wall20)
    walls.append(wall21)
    walls.append(wall22)
    walls.append(wall23)
    walls.append(wall24)
    walls.append(wall25)

    walls.append(wall27)
    walls.append(wall28)
    walls.append(wall29)
    walls.append(wall30)
    walls.append(wall31)
    walls.append(wall32)
    walls.append(wall33)
    walls.append(wall34)
    walls.append(wall35)
    walls.append(wall36)
    walls.append(wall37)
    walls.append(wall38)
    walls.append(wall39)
    walls.append(wall40)
    walls.append(wall41)
    walls.append(wall42)
    walls.append(wall43)
    walls.append(wall44)
    walls.append(wall45)
    walls.append(wall46)
    walls.append(wall47)
    walls.append(wall48)
    walls.append(wall49)
    walls.append(wall50)
    walls.append(wall51)
    walls.append(wall52)
    walls.append(wall53)
    walls.append(wall54)
    walls.append(wall55)
    walls.append(wall56)
    walls.append(wall57)
    walls.append(wall58)
    walls.append(wall59)
    walls.append(wall60)
    walls.append(wall61)
    walls.append(wall62)
    walls.append(wall63)
    walls.append(wall64)
    walls.append(wall65)
    walls.append(wall66)
    walls.append(wall67)
    walls.append(wall68)
    walls.append(wall69)
    walls.append(wall70)
    walls.append(wall71)
    walls.append(wall72)
    walls.append(wall73)
    walls.append(wall74)
    walls.append(wall75)
    walls.append(wall76)
    walls.append(wall77)
    walls.append(wall78)
    walls.append(wall79)
    walls.append(wall80)
    walls.append(wall81)
    walls.append(wall82)
    walls.append(wall83)
    walls.append(wall84)
    walls.append(wall85)
    walls.append(wall86)
    walls.append(wall87)
    walls.append(wall88)
    walls.append(wall89)
    walls.append(wall90)
    walls.append(wall91)
    walls.append(wall92)
    walls.append(wall93)
    walls.append(wall94)
    walls.append(wall95)
    walls.append(wall96)
    walls.append(wall97)
    walls.append(wall98)

    return (walls)